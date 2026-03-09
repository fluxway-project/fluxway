import unittest
from dataclasses import dataclass
import numpy as np

from fluxway.common.utils.tree import tree_flatten, tree_unflatten
from fluxway.common.decorators import data_type
from fluxway.common.types.tensor import TransportTensor


@data_type()
@dataclass
class Leaf:
    a: TransportTensor
    b: int


@data_type()
@dataclass
class Root:
    child: Leaf
    c: float


class TestTreeUtils(unittest.TestCase):
    def test_nested_data_type_flattening(self):
        t1 = TransportTensor(np.array([1, 2, 3]))
        root = Root(child=Leaf(a=t1, b=10), c=3.14)

        leaves, treedef = tree_flatten(root)

        self.assertEqual(len(leaves), 3)
        self.assertIs(leaves[0], t1)
        self.assertEqual(leaves[1], 10)
        self.assertEqual(leaves[2], 3.14)

        reconstructed = tree_unflatten(treedef, leaves)
        self.assertIsInstance(reconstructed, Root)
        self.assertIsInstance(reconstructed.child, Leaf)
        self.assertIs(reconstructed.child.a, t1)
        self.assertEqual(reconstructed.child.b, 10)
        self.assertEqual(reconstructed.c, 3.14)

    def test_guided_dict_flattening(self):
        # Normal sort order: ['a', 'b', 'c']
        # We want to guide it to: ['c', 'a', 'b']
        data = {"a": 1, "b": 2, "c": 3}
        template = ("dict", ["c", "a", "b"], [None, None, None])

        leaves, treedef = tree_flatten(data, treedef_template=template)

        self.assertEqual(leaves, [3, 1, 2])
        self.assertEqual(treedef[1], ["c", "a", "b"])

    def test_guided_nested_flattening(self):
        # Template: dict -> { x: [list(2)], y: data_type(Leaf) }
        @data_type()
        @dataclass
        class Simple:
            val: int

        data = {"x": [10, 20], "y": Simple(val=5)}

        # Template forces 'y' then 'x'
        simple_template = ("data_type", Simple, ["val"], [None])
        list_template = ("list", 2, [None, None])
        template = ("dict", ["y", "x"], [simple_template, list_template])

        leaves, treedef = tree_flatten(data, treedef_template=template)

        self.assertEqual(leaves, [5, 10, 20])
        self.assertEqual(treedef[1], ["y", "x"])
        self.assertEqual(treedef[2][0][0], "data_type")
        self.assertEqual(treedef[2][1][0], "list")

    def test_strict_key_mismatch_raises(self):
        # Template expects ['a', 'b'], but data has ['a', 'c']
        data = {"a": 1, "c": 3}
        template = ("dict", ["a", "b"], [None, None])

        with self.assertRaisesRegex(ValueError, "do not match declared sharding keys"):
            tree_flatten(data, treedef_template=template)


if __name__ == "__main__":
    unittest.main()
