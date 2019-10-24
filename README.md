#### A prototype of polio toy model:

* Simple S, I, R with constant population by village
* Connecting villages in explicit neighbor network
* Tracking individual infections -- simple fixed generation time + infectiousness
* Reporting summary statistics
* Reporting detailed transmission events

![Village Topology](/village_topology.png)

And some interactive plotly visualization to explore infection and immune dynamics:

![Spatial Epidemic Snapshot](/spatial_epidemic_snapshot.png)

And a randomly subsampled tree visualized with [`tskit`](https://tskit.readthedocs.io/en/stable/index.html)

![Sampled Tree Screenshot (tskit)](/tskit_sampled_tree_screenshot.png)
