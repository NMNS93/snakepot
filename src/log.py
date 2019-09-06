import sys
from logbook import Logger, NestedSetup, StreamHandler, FileHandler, StringFormatterHandlerMixin, NullHandler

format_string='[{record.time:%y%m%d %H:%M}] {record.level_name}: snakepot {record.channel}:  {record.message}'

NestedSetup([FileHandler('logfile.log', format_string=format_string, level='DEBUG'), 
             StreamHandler(sys.stderr, format_string=format_string, bubble=True)]).push_application()
