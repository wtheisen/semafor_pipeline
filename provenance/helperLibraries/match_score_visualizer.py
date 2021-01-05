import sys, json
import matplotlib.pyplot as plt

from os.path import basename

with open(sys.argv[1]) as f:
    data = json.load(f)

node_id_list = []
node_score_list = []

for node in data['nodes']:
    node_id_list.append(int(node['id']))
    node_score_list.append(float(node['nodeConfidenceScore']))

plt.plot(node_id_list, node_score_list, 'o', color='black')
plt.title(basename(sys.argv[1]))

plt.show()
