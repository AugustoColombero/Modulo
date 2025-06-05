# Excepcion contenedor vacía
class Empty(Exception):
    """ Empty container!"""
    pass


#
class _Node:
    def __init__(self, data, next):
        self._data = data
        self._next = next


class stack:
    # constructor        
    def __init__(self):
        self._head = None
        self._size = 0
        
    def __len__(self):
        return self._size
    
    #métodos 
    def is_empty(self):
        if self._size == 0:
            return True
        else:
            return False
            
    def push(self, data):
        if self._head == None:
            self._head = _Node(data, None)
        else:
            newnode = _Node(data, self._head)            
            self._head = newnode
        self._size += 1
    
    
    def pop(self):
        if self.is_empty():
            raise(Empty('Stack vacía'))
        answer = self._head._data
        self._head = self._head._next
        self._size -= 1
        return answer
    
    def top(self):
        if self.is_empty():
            raise(Empty('Stack vacía'))
        return self._head._data
    
    def __str__(self):
        node = self._head
        s = "["
        while not node is None:
            s += str(node._data) 
            if not node._next is None:
                s +=  ", "
            node = node._next
        s += "]"
        return s

    def destructive_print(self):
        L = []
        while len(self)>0:
            L.append(self.pop())
        print(L)                 
    
if __name__ == '__main__':
    s = stack()    
    for j in range(10):
        s.push(j)
    print(s)