"""

Perform manipulations as in tskit documentation:
https://tskit.readthedocs.io/en/stable/data-model.html#node-text-format

Manipulate forward-simulation outputs as in msprime documentation:
https://msprime.readthedocs.io/en/stable/tutorial.html#completing-forwards-simulations

"""


import os

import numpy as np
import pandas as pd
import tskit

# Read transmission tree
df = pd.read_csv(os.path.join('output', 'transmission.csv'))

# Sub-sample
df['is_sample'] = np.random.random(size=len(df)) < 1/200.  # AFP
ES_slice = df.loc[df.village.isin([136, 163])]
df.loc[ES_slice.index, 'is_sample'] = np.random.random(size=len(ES_slice)) < 0.1  # ES
df.loc[pd.isnull(df.transmitter), 'is_sample'] = False  # ensure initial infections not sampled?
df.loc[pd.isnull(df.transmitter), 'generation'] = -1  # tskit requires time[parent] > time[child]
# print(df[df.is_sample].head(20))

# Dump text files of nodes + edges in format expected by tskit.load_text
df['time'] = df.generation.max() - df.generation  # coalescent models count time backward from present
df['is_sample'] = df.is_sample.astype(int)  # expects 0,1 not False, True
df[['is_sample', 'time']].to_csv(os.path.join('output', 'node_table.txt'), index=False, sep='\t')

df['left'] = 0  # using the whole genome as the transmitted unit (i.e. no recombination)
df['right'] = 1
edges = df[['transmitter', 'infected', 'left', 'right']]
edges.columns = ['parent', 'child', 'left', 'right']
valid_edges = edges[pd.notnull(edges.parent)].copy()  # exclude "orphan" edges from initial infections
valid_edges['parent'] = valid_edges.parent.astype(int)
valid_edges.to_csv(os.path.join('output', 'edge_table.txt'), index=False, sep='\t')

# Load back into tskit.TreeCollection
with open(os.path.join('output', 'node_table.txt')) as nodes:
    with open(os.path.join('output', 'edge_table.txt')) as edges:
        tree_sequence = tskit.load_text(nodes=nodes, edges=edges)

# Let's see what we did?
print(f"The tree sequence has {tree_sequence.num_trees} trees on a genome of length {tree_sequence.sequence_length},"
      f" {tree_sequence.num_individuals} individuals, {tree_sequence.num_samples} 'sample' genomes,"
      f" and {tree_sequence.num_mutations} mutations.")
tree = tree_sequence.first()
# print(tree.draw(format='unicode'))  # don't even think about it!
print('Tree nodes before pruning unsampled lineages: %d' % tree.num_nodes)
sampled_ts = tree_sequence.simplify()
sampled_tree = sampled_ts.first()
print('Tree nodes after pruning unsampled lineages: %d' % sampled_tree.num_nodes)
# print(sampled_tree.draw(format='unicode'))  # almost visualizable, but not quite
with open(os.path.join('output', 'sampled_tree.txt'), 'w') as f:
    f.write(sampled_tree.draw(format='unicode'))  # just write to file + use really small font

# Dump binary formatted .trees

# Load binary .trees to tskit.TreeCollection

# ------------

# Alternatively, build TreeCollection dynamically from TableCollection API (i.e. add_node, add_row)

# ------------

# re-run mutations

# generate genomes
