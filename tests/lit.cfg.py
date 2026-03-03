import lit.formats
import os

config.name = "IOOptimisationPass Suite"
config.test_format = lit.formats.ShTest(True)
config.suffixes = ['.c', '.cpp']

config.environment['PATH'] = os.environ.get('PATH', '')
if 'CPATH' in os.environ:
    config.environment['CPATH'] = os.environ.get('CPATH')

if hasattr(config, 'shlibdir'):
    config.substitutions.append(('%shlibdir', config.shlibdir))
if hasattr(config, 'shlibext'):
    config.substitutions.append(('%shlibext', config.shlibext))
