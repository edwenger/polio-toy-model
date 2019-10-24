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


output_directory = 'output'


def parse_transmission_events():
    df = pd.read_csv(os.path.join('output', 'transmission.csv'))
    df.loc[pd.isnull(df.transmitter), 'generation'] = -1  # tskit requires time[parent] > time[child]
    df['time'] = df.generation.max() - df.generation  # coalescent models count time backward from present
    return df


def transmission_sampled(df, afp_rate=1/200., es_rate=0.1, es_villages=[]):
    sample_s = np.random.random(size=len(df)) < afp_rate  # AFP
    es_slice = df.loc[df.village.isin(es_villages)]
    sample_s[es_slice.index] = np.random.random(size=len(es_slice)) < es_rate  # ES
    sample_s[pd.isnull(df.transmitter)] = False  # ensure initial infections not sampled?
    return sample_s


def dump_nodes(df, sample_s, path=os.path.join(output_directory, 'node_table.txt')):
    df['is_sample'] = sample_s.astype(int)  # tskit.load_text expects 0,1 not False, True
    df[['is_sample', 'time']].to_csv(path, index=False, sep='\t')


def populate_node_table(tc, df, sample_s):

    df['is_sample'] = sample_s.astype(int)  # tskit.load_text expects 0,1 not False, True

    for _, row in df.iterrows():
        tc.nodes.add_row(flags=int(row.is_sample), time=float(row.time))  # IS_SAMPLE = 1 in bitmask


def dump_edges(df, path=os.path.join(output_directory, 'edge_table.txt')):

    valid_edges = df[pd.notnull(df.transmitter)].copy()  # exclude "orphan" edges from initial infections

    # reshape to names and types expected by tskit.load_text
    valid_edges = valid_edges[['transmitter', 'infected']]
    valid_edges.columns = ['parent', 'child']
    valid_edges['parent'] = valid_edges.parent.astype(int)

    # using the whole genome as the transmitted unit (i.e. no recombination)
    valid_edges['left'] = 0
    valid_edges['right'] = 1

    valid_edges.to_csv(path, index=False, sep='\t')


def populate_edge_table(tc, df):

    for _, row in df[pd.notnull(df.transmitter)].iterrows():  # exclude "orphan" edges
        tc.edges.add_row(left=0.0, right=1.0, parent=int(row.transmitter), child=int(row.infected))


def load_tree_collection(node_path=os.path.join(output_directory, 'node_table.txt'),
                         edge_path=os.path.join(output_directory, 'edge_table.txt')):
    with open(node_path) as nodes:
        with open(edge_path) as edges:
            ts = tskit.load_text(nodes=nodes, edges=edges)

    return ts


def simplify_tree(ts):

    tree = ts.first()
    # print(tree.draw(format='unicode'))  # don't even think about it!
    print('Tree nodes before pruning unsampled lineages: %d' % tree.num_nodes)

    simplified_ts = ts.simplify()
    simplified_tree = simplified_ts.first()
    # print(simplified_tree.draw(format='unicode'))  # almost visualizable, but not quite
    print('Tree nodes after pruning unsampled lineages: %d' % simplified_tree.num_nodes)

    return simplified_ts


def inspect_tree_collection(ts):

    print(f"The tree sequence has {ts.num_trees} trees on a genome of length {ts.sequence_length},"
          f" {ts.num_individuals} individuals, {ts.num_samples} 'sample' genomes,"
          f" and {ts.num_mutations} mutations.")


def draw_tree_to_file(ts, path=os.path.join(output_directory, 'sampled_tree.txt')):
    tree = ts.first()
    with open(path, 'w') as f:
        f.write(tree.draw(format='unicode'))


def dump_binary_tree(ts, path=os.path.join(output_directory, 'sampled_tree_sequence.trees')):
    ts.dump(path)


def load_binary_tree(path=os.path.join(output_directory, 'sampled_tree_sequence.trees')):
    return tskit.load(path)


if __name__ == '__main__':

    # get the output of the toy model
    transmissions_df = parse_transmission_events()

    # impose AFP + ES random sampling
    is_sampled = transmission_sampled(transmissions_df, es_villages=[136, 163])
    # print(transmissions_df[is_sampled].head(20))

    # ------------
    # We can write out files and read them back in as tskit.TreeCollection

    # # dump to file
    # dump_nodes(transmissions_df, is_sampled)
    # dump_edges(transmissions_df)
    #
    # # load back from file as tskit.TreeCollection
    # tree_sequence = load_tree_collection()

    # ------------
    # Alternatively, we can build it dynamically using the TableCollection API
    # (i.e. add_node, add_row)

    table_collection = tskit.TableCollection(sequence_length=1.0)
    populate_node_table(table_collection, transmissions_df, is_sampled)
    populate_edge_table(table_collection, transmissions_df)

    table_collection.sort()  # requires (time[parent], child, left) order
    tree_sequence = table_collection.tree_sequence()

    # ------------
    # Let's see what we did?

    # inspect_tree_collection(tree_sequence)

    # Simplify tree based down to sampled lineages
    sampled_ts = simplify_tree(tree_sequence)

    # Draw ASCII tree to file + view with really tiny font
    draw_tree_to_file(sampled_ts)

    # Dump to binary tree format
    dump_binary_tree(sampled_ts)

    # Reload binary file to tskit.TreeCollection
    # reloaded_ts = load_binary_tree()
    # reloaded_tree = reloaded_ts.first()
    # print('Reloaded tree has nodes: %d' % reloaded_tree.num_nodes)

    # ------------
    # Re-run mutations

    # generate genomes
