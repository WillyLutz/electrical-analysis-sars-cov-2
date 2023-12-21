from scripts.pipelines import confusion, pca

merge_stachel ="example_data/merge stachel.csv"

merge_vlp_spike = "example_data/merge vlp+ vlp- spike.csv"

# fig 4I
confusion(["-stachel ni t24", "-stachel inf t24"], ["-stachel stachel inf t24",], merge_path=merge_stachel,)

# fig S8D
confusion(["Mock T24", "Inf T24"], ["Spike T24",], merge_path=merge_vlp_spike,)

# fig S8C
confusion(["-stachel ni t24", "-stachel inf t24"], ["-stachel ni t0", "-stachel ni t30", "-stachel inf t0", "-stachel inf t30"], merge_path=merge_stachel,)


