"""

Perform manipulations as in tskit documentation:
https://tskit.readthedocs.io/en/stable/data-model.html#node-text-format

Manipulate forward-simulation outputs as in msprime documentation:
https://msprime.readthedocs.io/en/stable/tutorial.html#completing-forwards-simulations

"""


import os
from collections import defaultdict

import numpy as np
import pandas as pd
import tskit
import msprime


output_directory = 'output'

genome_length = 300

N = 100  # historical haploid effective population


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
    valid_edges['right'] = genome_length

    valid_edges.to_csv(path, index=False, sep='\t')


def populate_edge_table(tc, df):

    for _, row in df[pd.notnull(df.transmitter)].iterrows():  # exclude "orphan" edges
        tc.edges.add_row(left=0, right=genome_length, parent=int(row.transmitter), child=int(row.infected))


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


def dump_tree_to_newick(ts, path=os.path.join(output_directory, 'recapitated_newick.txt')):
    if ts.num_trees != 1:
        raise Exception('Expecting only to write a single tree.')
    tree = ts.first()
    with open(path, 'w') as f:
        f.write(tree.newick())


def dump_binary_tree(ts, path=os.path.join(output_directory, 'sampled_tree_sequence.trees')):
    ts.dump(path)


def load_binary_tree(path=os.path.join(output_directory, 'sampled_tree_sequence.trees')):
    return tskit.load(path)


def inspect_degenerate_mutations(ts):

    mutations_by_node_position = defaultdict(int)  # count mutations by (node, integer_genome_position)

    if ts.num_trees != 1:
        raise Exception('Not yet implemented loop over trees')

    tree = ts.first()

    for site in tree.sites():
        if len(site.mutations) != 1:
            raise Exception('How in infinite-site mode?!')
        for mutation in site.mutations:
            # print('Mutation @ position {:.2f} over node {} with parent {}'.format(site.position, mutation.node))
            mutations_by_node_position[(mutation.node, int(site.position))] += 1

    mutations_df = pd.DataFrame.from_dict(mutations_by_node_position, orient='index')
    mutations_df.columns = ['n_mutations']
    mutations_df.reset_index(inplace=True)
    mutations_df[['node', 'site']] = pd.DataFrame(mutations_df['index'].tolist(), index=mutations_df.index)
    n_mutations = mutations_df.set_index(['site', 'node']).n_mutations
    odd_mutations = n_mutations[n_mutations % 2 == 1].sort_values(ascending=False)
    # print(odd_mutations.head(20))

    return odd_mutations


def squash_finite_site_mutations(ts, odd_mutations):

    """
    Remake SiteTable and MutationTable
    based on non-redundant (i.e. odd-numbered) mutations on each edge
    by recursively traversing full transmission tree.

    Note requirements: https://tskit.readthedocs.io/en/latest/data-model.html#mutation-table
    """

    print('Original TreeSequence with %d sites and %d mutations' % (ts.num_sites, ts.num_mutations))

    tables = ts.tables  # mutable copy of tree sequence

    # clear sites + mutations, which we'll discretize + squash
    tables.sites.clear()
    tables.mutations.clear()

    if ts.num_trees != 1:
        raise Exception('Not yet implemented loop over trees')

    tree = ts.first()

    if len(tree.roots) != 1:
        raise Exception('Not yet implemented loop over roots')

    site_ids = {}

    def iter_tree(node, mutation_by_parent):

        # get sites of mutations for node
        mutations = odd_mutations[odd_mutations.node == node].site.values

        # get state of ancestor and record updated state
        for position in mutations:

            if position not in site_ids:
                site_id = tables.sites.add_row(position=position, ancestral_state='0')
                site_ids[position] = site_id
                # print('Site: position=%d, ancestral_state=%s' % (position, '0'))

            parent_state_tuple = mutation_by_parent.get(position, None)

            if parent_state_tuple is None:
                parent = -1  # NULL parent
                child_state = '1'
            else:
                parent, parent_state = parent_state_tuple
                child_state = '1' if parent_state == '0' else '0'

            site = site_ids[position]  # ID of position in SiteTable

            mutation_id = tables.mutations.add_row(site=site, node=node, parent=parent, derived_state=child_state)
            # print('Mutation: site=%d, node=%d, parent=%d, derived_state=%s' % (site, node, parent, child_state))
            mutation_by_parent[position] = (mutation_id, child_state)  # pass updated mutation info to children

        if tree.is_leaf(node):
            return  # dead end
        else:
            for child in tree.children(node):
                iter_tree(child, mutation_by_parent.copy())  # pass copy to not overwrite

    iter_tree(tree.root, {})  # begin iteration

    tables.sort()
    squashed = tables.tree_sequence()
    print('Squashed TreeSequence with %d sites and %d mutations' % (squashed.num_sites, squashed.num_mutations))

    return squashed


def inspect_variants(ts):
    """ Look at sample genotypes by variant """

    for variant in ts.variants():
        print(variant.site.id, variant.site.position, variant.alleles, variant.genotypes, sep='\t')


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

    tables = tskit.TableCollection(sequence_length=genome_length)
    populate_node_table(tables, transmissions_df, is_sampled)
    populate_edge_table(tables, transmissions_df)

    # At least one population for recapitation
    tables.populations.add_row()
    tables.nodes.set_columns(
        flags=tables.nodes.flags,
        time=tables.nodes.time,
        population=np.zeros_like(tables.nodes.population))

    tables.sort()  # requires (time[parent], child, left) order
    tree_sequence = tables.tree_sequence()

    # ------------
    # Let's see what we did?

    # inspect_tree_collection(tree_sequence)

    # Simplify tree based down to sampled lineages
    sampled_ts = simplify_tree(tree_sequence)

    # Draw ASCII tree to file + view with really tiny font
    # draw_tree_to_file(sampled_ts)

    # Dump to binary tree format
    # dump_binary_tree(sampled_ts)

    # Reload binary file to tskit.TreeCollection
    # reloaded_ts = load_binary_tree()
    # reloaded_tree = reloaded_ts.first()
    # print('Reloaded tree has nodes: %d' % reloaded_tree.num_nodes)

    # ------------
    # Recapitate -- run coalescent model on un-coalesced trees from forward-simulation model

    uncoalesced_trees = sum([t.num_roots > 1 for t in sampled_ts.trees()])
    print('There are %d of %d uncoalesced trees' % (uncoalesced_trees, sampled_ts.num_trees))

    # We are simulating haploid, but msprime Ne is diploid Wright-Fischer, so use N/2
    coalesced_ts = msprime.simulate(Ne=N/2, from_ts=sampled_ts)

    # draw_tree_to_file(coalesced_ts, path=os.path.join(output_directory, 'recapitated_tree.txt'))

    # Write to NEWICK format
    # dump_tree_to_newick(coalesced_ts)

    # ------------
    # Run a mutation model over our full tree to generate genomes

    mutated_ts = msprime.mutate(coalesced_ts, rate=0.1)  # mutations (in genome of interest) per generation

    # Currently only InfiniteSiteMutation model, but open issue with some traction to implement finite genome:
    # https://github.com/tskit-dev/msprime/issues/553

    print(f"The tree sequence now has {mutated_ts.num_trees} trees,"
          f" and {mutated_ts.num_mutations} mutations.")

    odd_count_mutations = inspect_degenerate_mutations(mutated_ts)

    squashed_ts = squash_finite_site_mutations(mutated_ts, odd_count_mutations.reset_index())

    # inspect_variants(squashed_ts)

    # with open(os.path.join(output_directory, 'mutated_genomes.fasta'), 'w') as fasta_file:
    #     squashed_ts.write_fasta(fasta_file)  # only in >=0.2.3dev0 (not in 0.2.2)

    with open(os.path.join(output_directory, 'mutated_genomes.vcf'), 'w') as vcf_file:
        squashed_ts.write_vcf(vcf_file)