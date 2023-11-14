from scripts.pipelines import confusion, pca, feature_importance

feature_importance(["Mock T24", "Inf T24"], )
pca(["Mock T24", "Inf T24"], ["Spike T24",], 2)
confusion(["Mock T24", "Inf T24"], ["Spike T24",])
