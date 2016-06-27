import unittest

from LinkedList import LinkedList


class TestLinkedList(unittest.TestCase):

    def setUp(self):
        n = 5
        a = list(range(1, n + 1))
        b = [0 for x in range(1, n + 1)]
        strs = ['a', 'b', 'c']

        # Make lists
        x = LinkedList()
        y = LinkedList(a)
        z = LinkedList(b)
        c = LinkedList(strs)

        self.contents = [[], a, b, strs]
        self.sizes = [0, n, n, 3]
        self.lists = [x, y, z, c]

    def test_listLengths(self):
        for i in range(len(self.lists)):
            lst = self.lists[i]
            size = self.sizes[i]
            self.assertEqual(lst.size, size)

    def test_listContents(self):
        for i in range(len(self.lists)):
            lst = self.lists[i]
            contents = self.contents[i]
            self.assertEqual(list(lst), contents)

    def test_get(self):
        x = self.lists[1]
        x_ = self.contents[1]
        self.assertEqual(x.get(), x_[-1])
        self.assertEqual(x.get(0), x_[0])

    def test_set(self):
        x = self.lists[1]
        x_ = self.contents[1]
        x.set(0, 99)
        x_[0] = 99
        x.set(-1, 100)
        x_[-1] = 100
        self.assertEqual(list(x), x_)

    def test_append(self):
        x = self.lists[1]
        before_contents = list(x)
        compare_contents = before_contents
        x.append(99)
        x.append(100)
        compare_contents.append(99)
        compare_contents.append(100)
        after_contents = list(x)
        self.assertEqual(after_contents, compare_contents)

        # Make sure append treats a string as a single entry
        y = self.lists[3]
        before_contents = list(y)
        compare_contents = before_contents
        y.append('xyz')
        compare_contents.append('xyz')
        after_contents = list(y)
        self.assertEqual(after_contents, compare_contents)

    def test_extend(self):
        x = self.lists[1]
        before_contents = list(x)
        compare_contents = before_contents
        x.extend([99,100])
        compare_contents.extend([99,100])
        after_contents = list(x)
        self.assertEqual(after_contents, compare_contents)

        y = self.lists[3]
        before_contents = list(y)
        compare_contents = before_contents
        y.extend(['xyz', '123'])
        compare_contents.extend(['xyz', '123'])
        after_contents = list(y)
        self.assertEqual(after_contents, compare_contents)

    def test_extendLinkedList(self):
        x = self.lists[1]
        y = self.lists[2]
        before_contents = list(x)
        compare_contents = before_contents
        x.extend(y)
        compare_contents.extend(list(y))
        after_contents = list(x)
        self.assertEqual(after_contents, compare_contents)

    def test_prepend(self):
        x = self.lists[1]
        before_contents = list(x)
        x.prepend(99)
        x.prepend(100)
        compare_contents = [100, 99]
        compare_contents.extend(before_contents)
        after_contents = list(x)
        self.assertEqual(after_contents, compare_contents)

    def test_delete(self):
        x = LinkedList([1,2,3,4,5])
        val1 = x.delete(0)  # 1st element
        val2 = x.delete(-1)  # last element
        self.assertEqual(val1, 1)
        self.assertEqual(val2, 5)
        self.assertEqual(list(x), [2,3,4])

    def test_check_bounds(self):
        x = self.lists[1]
        self.assertRaises(IndexError, x.get, 20)  # index out of bounds

    def test_default_fwd_iter(self):
        x = self.lists[1]
        x_ = self.contents[1]
        x_iter = x.get_iterator()
        for i in range(len(x_)):
            self.assertEqual(x_iter.next(), x_[i])

    def test_fwd_iter(self):
        x = self.lists[1]
        x_ = self.contents[1]
        x_iter = x.get_iterator(0)
        self.assertEqual(x_iter.value(), x_[0])
        for i in x_[1:len(x_)]:
            self.assertEqual(x_iter.next(), i)

    def test_rev_iter(self):
        x = self.lists[1]
        x_ = self.contents[1]
        x_iter = x.get_iterator(-1)
        self.assertEqual(x_iter.value(), x_[-1])
        for i in x_[-2::-1]:
            self.assertEqual(x_iter.prev(), i)

    def test_mutating_iter(self):
        x = LinkedList([1,2,3,4,5])
        x_iter = x.get_iterator()
        x_iter.next()
        val = x_iter.next()
        self.assertEqual(val, 2)
        x_iter.insert(99)
        x_iter.insert(100)
        x_iter.next()
        x_iter.insert(101)
        val = x_iter.next()
        self.assertEqual(val, 4)
        val = x_iter.delete()
        self.assertEqual(val, 4)
        x_array = list(x)
        self.assertEqual(x_array, [1,2,99,100,3,101,5])

if __name__ == '__main__':
    unittest.main()
