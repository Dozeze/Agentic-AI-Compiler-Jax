import os
import pprint

# Pretty-print all environment variables as a dictionary
pprint.pprint(dict(os.environ))

# Look for a specific variable safely
print(os.environ.get("USER"))  # Returns None if not found
