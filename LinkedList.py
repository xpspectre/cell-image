import math


def is_int(value):
    """Tests whether value is an integer value (numeric type that represents an integer value)"""
    return value == math.floor(value)


def check_index_bounds(index, size):
    """Validates whether index is an integer and in the list of size. Throws IndexError exception if not."""
    if size == 0:
        raise IndexError('Size of list is 0 so no indexing is valid')

    if index >= 0:
        if index >= size:
            raise IndexError(
                'Index {index} out of bounds for list of length {length}'.format(index=index, length=size))
    else:  # index is negative
        if -index > size:
            raise IndexError(
                'Negative index {index} out of bounds for list of length {length}'.format(index=index, length=size))


def fix_index(index, size):
    """Convert negative index to positive index if needed"""
    if index < 0:
        return size + index
    else:
        return index


class LinkedList:
    """Doubly-linked list. Indexing starts at 0.
    Can contain empty (value = None) nodes.
    Negative indices count backwards from the end"""

    def __init__(self, values=[]):
        """Construct linnked list
        Usage:
            ll = LinkedList(): Constrct an empty list
            ll = LinkedList(values): Construct a list containing the
               elements from the array, cell array, or LinkedList `values`,
                keeping their order. Copies the values of the LinkedList.

        Note: Use this to directly construct a list from a Python list (array with
            O(1) time lookups) or other LinkedList (with O(n) time lookups).
        """
        self.size = 0
        self.head = Node()
        self.tail = Node()
        self.head.next = self.tail
        self.tail.prev = self.head
        self.extend(values)

    def get(self, index=None):
        """Get `value` in list at `index`. If `index` not specified, get last value.
        Warning: Runs in O(n) time for arbitrary elements in the middle."""
        if index is None:
            index = self.size - 1
        check_index_bounds(index, self.size)
        index = fix_index(index, self.size)
        node = self.get_node(index)
        return node.value

    def set(self, index, value):
        """Replace `value` at `index`. Can only replace values, i.e., cannot add a value. Returns nothing.
        Warning: Runs in O(n) time for arbitrary elements in the middle."""
        check_index_bounds(index, self.size)
        index = fix_index(index, self.size)
        node = self.get_node(index)
        node.value = value

    def append(self, value):
        """Append single value (contents of value treated as single value) to end of list"""
        self.insert([value], self.size)

    def extend(self, values):
        """Extend list with entries in iterable values. Includes other Linked Lists, which are iterated through in the
        forward direction"""
        self.insert(values, self.size)

    def prepend(self, value):
        """Prepend single value (contents of value treated as a single value) to beginning of list"""
        self.insert([value], 0)

    def insert(self, values, index):
        """Insert values in middle of list. index is the position the (1st of the) new values will be.
         If values is iterable, adds each element individually. Otherwise, the single value is added.
         Warning: O(n) lookup to find arbitrary position in middle of list.
         This method is used as the basis of the other methods that add values to the list.
         append and prepend wrap their single value inside a list, which has 1 entry, and is inserted in 1 operation."""

        if not is_int(index):
            raise IndexError('Index {index} not an integer'.format(index=index))

        check_index_bounds(index, self.size + 1)  # +1 accounts for extra position from inserting
        index = fix_index(index, self.size)

        # Go to index where insertion will happen
        node = self.get_node(index)

        # Get nodes to insert new nodes next to
        lnode = node.prev
        rnode = node

        # Insert new value(s)
        size = 0
        if hasattr(values, '__iter__') and not isinstance(values, str):  # iterable and not string
            for value in values:
                size += 1
                newnode = Node(value, lnode, rnode)
                lnode.next = newnode
                rnode.prev = newnode
                lnode = newnode
        else:  # single value
            newnode = Node(values, lnode, rnode)
            lnode.next = newnode
            rnode.prev = newnode

        self.size += size

    def delete(self, index):
        """Delete value at `index`. Returns the `value` deleted."""
        check_index_bounds(index, self.size)
        index = fix_index(index, self.size)

        node = self.get_node(index)
        value = node.value

        lnode = node.prev
        rnode = node.next
        lnode.next = rnode
        rnode.prev = lnode
        self.size -= 1

        return value

    def __iter__(self):
        return self.get_iterator()

    def get_iterator(self, index=None):
        """Get an iterator (actual a more general "traverser") for the list. `index` is the element that the iterator
        starts on, defaulting to `this.head` (call `iterator.next` to get the first value). Note that the specifying
        an index will place the iterator on that element and calling iter.next or iter.prev will give you the 'next'
        element in that direction."""
        if index is None:
            return Iterator(self, self.head)

        check_index_bounds(index, self.size)
        index = fix_index(index, self.size)
        node = self.get_node(index)
        return Iterator(self, node)

    def get_node(self, index):
        """Get the node at `index`. Starts from the head or tail, whichever is closer. Warning: runs in O(n) time for
        arbitrary but O(n) to get first or last node."""
        check_index_bounds(index, self.size + 1)  # picking the right sentinel node is OK
        index = fix_index(index, self.size)
        if index < self.size / 2:  # closer to head
            offset = index + 1  # the number of nodes to jump
            node = self.head
            for i in range(offset):
                node = node.next
        else:  # closer to tail
            offset = self.size - index
            node = self.tail
            for i in range(offset):
                node = node.prev

        return node


class Iterator():
    """General purpose linked list iterator/traverser
    Note: iter.value() returns the current value of the iterator but iter.next() and  iter.prev() return the next and
    previous values in addition to moving the iterator. Maybe change this."""

    def __init__(self, list, node):
        self.list = list
        self.node = node

    def __next__(self):
        """Usual Python method to make this a (forward) iterator"""
        return self.next()

    def next(self):
        if not self.has_next():
            raise StopIteration
        self.node = self.node.next
        return self.node.value

    def prev(self):
        if not self.has_prev():
            raise StopIteration
        self.node = self.node.prev
        return self.node.value

    def has_next(self):
        nextnode = self.node.next
        if not nextnode.next:
            return False
        else:
            return True

    def has_prev(self):
        prevnode = self.node.prev
        if not prevnode.prev:
            return False
        else:
            return True

    def value(self):
        return self.node.value

    def insert(self, value):
        """Insert value to the right of current position and set iterator to the new value/position"""
        lnode = self.node
        rnode = self.node.next
        newnode = Node(value, lnode, rnode)
        lnode.next = newnode
        rnode.prev = newnode
        self.node = newnode
        self.list.size += 1

    def delete(self):
        value = self.node.value
        lnode = self.node.prev
        rnode = self.node.next
        lnode.next = rnode
        rnode.prev = lnode
        self.list.size -= 1
        return value


class Node:

    def __init__(self, value=None, prev=None, next=None):
        self.value = value
        self.prev = prev
        self.next = next