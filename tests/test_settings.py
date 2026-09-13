from data.settings import AppSettings, SettingsStore


class MemorySettings:
    def __init__(self, values=None):
        self.values = dict(values or {})

    def value(self, key, default=None):
        return self.values.get(key, default)

    def setValue(self, key, value):
        self.values[key] = value

    def remove(self, prefix):
        for key in list(self.values):
            if key == prefix or key.startswith(f"{prefix}/"):
                del self.values[key]

    def sync(self):
        pass


def test_settings_round_trip():
    backend = MemorySettings()
    store = SettingsStore(backend)
    expected = AppSettings().updated(
        transect_num_lines=24,
        clahe_clip_limit=3.5,
        morphology_radius=4,
    )

    store.save(expected)

    assert store.load() == expected


def test_invalid_persisted_values_fall_back_to_defaults():
    backend = MemorySettings({
        "preferences/transect_num_lines": "not-a-number",
        "preferences/clahe_tile_size": -4,
        "preferences/minimum_typical_um_per_px": 50,
        "preferences/maximum_typical_um_per_px": 1,
    })

    loaded = SettingsStore(backend).load()

    assert loaded.transect_num_lines == AppSettings().transect_num_lines
    assert loaded.clahe_tile_size == AppSettings().clahe_tile_size
    assert loaded.minimum_typical_um_per_px == AppSettings().minimum_typical_um_per_px
    assert loaded.maximum_typical_um_per_px == AppSettings().maximum_typical_um_per_px


def test_reset_removes_persisted_values():
    backend = MemorySettings()
    store = SettingsStore(backend)
    store.save(AppSettings().updated(transect_num_lines=99))

    defaults = store.reset()

    assert defaults == AppSettings()
    assert store.load() == AppSettings()
