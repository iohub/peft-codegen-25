import os, sys
import pickle
import json



infile = sys.argv[1]
outfile = sys.argv[2]

infile = open(infile, 'rb')
outfile = open(outfile, 'w')
data = pickle.load(infile)
print(len(data))
cnt = 0
for entry in data:
    docstring_summary = entry['docstring_summary']
    docstring_summary = docstring_summary.replace('\n', '\\n')
    function = entry['function']
    function = function.replace('\n', '\\n')
    text = function + '\n'
    if docstring_summary:
        text = '// ' + docstring_summary + function + '\n'
    codeline = {'text': text}
    outfile.write( json.dumps(codeline) + ',\n' )
    cnt += 1
    # if cnt > 20: break

outfile.close()
infile.close()