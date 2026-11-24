from itertools import compress

import torch


class ObjDescription:
    def __init__(self, object_descriptions):
        self.data = object_descriptions

    def __getitem__(self, item):
        assert isinstance(item, torch.Tensor)
        assert item.dim() == 1
        if len(item) > 0:
            assert item.dtype == torch.int64 or item.dtype == torch.bool
            if item.dtype == torch.int64:
                return ObjDescription([self.data[x.item()] for x in item])
            if item.dtype == torch.bool:
                return ObjDescription(list(compress(self.data, item)))

        return ObjDescription(list(compress(self.data, item)))

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        return f"ObjDescription({self.data})"
