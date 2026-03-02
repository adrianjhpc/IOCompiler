import lit.formats

config.name = "IOOptimisationPass Suite"
# Use the shell test format (allows us to run pipeline commands)
config.test_format = lit.formats.ShTest(True)
# Look for files ending in .c and .cpp
config.suffixes = ['.c','.cpp']
