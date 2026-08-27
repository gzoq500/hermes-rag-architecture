from rag_gateway.schema import ensure_collection


class FakeClient:
    def __init__(self, exists):
        self.exists = exists
        self.calls = []

    def has_collection(self, **kwargs):
        self.calls.append(("has", kwargs))
        return self.exists

    def drop_collection(self, **kwargs):
        self.calls.append(("drop", kwargs))

    def create_collection(self, **kwargs):
        self.calls.append(("create", kwargs))

    @staticmethod
    def create_schema(**kwargs):
        return FakeSchema()

    @staticmethod
    def prepare_index_params():
        return FakeIndex()


class FakeSchema:
    def __init__(self):
        self.fields = []

    def add_field(self, **kwargs):
        self.fields.append(kwargs)


class FakeIndex:
    def __init__(self):
        self.indexes = []

    def add_index(self, **kwargs):
        self.indexes.append(kwargs)


def test_existing_collection_is_non_destructive_by_default():
    client = FakeClient(exists=True)

    result = ensure_collection(client, "memory", 3072, reset=False)

    assert result == "unchanged"
    assert [name for name, _ in client.calls] == ["has"]


def test_explicit_reset_drops_and_recreates_existing_collection():
    client = FakeClient(exists=True)

    result = ensure_collection(client, "memory", 3072, reset=True, data_type=object())

    assert result == "reset"
    assert [name for name, _ in client.calls] == ["has", "drop", "create"]


def test_missing_collection_is_created():
    client = FakeClient(exists=False)

    result = ensure_collection(client, "memory", 3072, reset=False, data_type=object())

    assert result == "created"
    assert [name for name, _ in client.calls] == ["has", "create"]
