"""
The goal of this coding activity is to design a system that limits the number of active roles that any given person has. A role gives the user access to some thing, whether it be a piece of data or an internal system. The system a used, the oldest role is removed if there are already k active rchieves this requirement by keeping track of the last k roles that a person has used. If a new role isoles for that person. Each role has a name and a message which contains details about its use by the person. You only need to store the last message for a role invocation.

Implement the constructor, get, and set methods of RolesCache. Each instance of the RolesCache corresponds to a single person.

Finally, fill out the runtime complexity for get and set and the overall space used. Use Big O notation, i.e. O(1), O(N), etc. For a refresher on Big O notation, please review https://danielmiessler.com/study/big-o-notation/.

"""

class RolesCache:
    def __init__(self, capacity):
        self.capacity = capacity
        self.cache = {}  # Dictionary to store roles and messages
        self.order = deque()  # Queue to keep track of the order of role invocations


    def get(self, role):
        return self.cache.get(role)  # Returns the message corresponding to the role, None if role does not exist

    def set(self, role, message):
        if role in self.cache:
            self.cache[role] = message  # Update message if role already exists
        else:
            if len(self.order) == self.capacity:
                oldest_role = self.order.popleft()  # Remove oldest role if cache is full
                del self.cache[oldest_role]
            self.cache[role] = message
            self.order.append(role)  # Add role to the end of the order queue

    def _complexity(self):
        return {
            'get': 'O(1)',  # Complexity of dictionary lookup
            'set': 'O(1)',  # Complexity of dictionary insertion and queue operations
            'space': 'O(N)'  # Space used by the dictionary and queue
        }