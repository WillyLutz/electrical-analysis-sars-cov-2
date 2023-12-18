from scripts.pipelines import confusion, pca

merge_stachel = "example_data/merge stachel.csv"
merge_vlp_spike = "example_data/merge vlp+ vlp- spike.csv"

confusion(["-stachel ni t24", "-stachel inf t24"], ["-stachel stachel inf t24"], merge_path=merge_stachel)

# confusion(["Mock T24", "Inf T24"], ["Spike T24",], merge_path=merge_vlp_spike)

# pca(["Mock T24", "Inf T24"], ["Spike T24"], 2, merge_path=merge_vlp_spike)