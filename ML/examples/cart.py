import numpy as np

def gini(y):
    c = np.bincount(y)
    pi = c / len(y)
    return 1 - np.sum(pi**2)

def build_cart_tree(X, y, max_depth=3, min_samples_split=2, depth=0):
    # print("AAAAAA")
    if len(y) < min_samples_split or depth >= max_depth or len(np.unique(y)) == 1: 
        return {"leaf": True,
            "prediction": float(np.bincount(y).argmax())}
            
    n, d = X.shape
    # print(n,d)
    
    best_threshold, best_feature, best_gini = None, None, gini(y)

    for feature in range(d):
        values = np.sort(np.unique(X[:, feature]))
        for i in range(len(values) - 1):
                threshold = float((values[i] + values[i+1]) / 2)
                # print("THRESHOLD", threshold, feature)
                # print("zzzz")
                
                mask = X[:, feature] <= threshold
                left_X, left_y = X[mask],y[mask]
                right_X, right_y = X[~mask],y[~mask]

                # print("BBB")
                left_gini = gini(left_y)
                # print(left_gini)
                right_gini = gini(right_y)
                # print(right_gini)

                # splitting leads to a net gain
                weighted_gini = (
                    len(left_y) / n * gini(left_y)
                    + len(right_y) / n * gini(right_y)
                )
                if weighted_gini < best_gini:
                    best_threshold, best_feature, best_gini = threshold, feature, weighted_gini

    if best_feature is None:
        return {"leaf": True,
                "prediction": float(np.bincount(y).argmax())}

    best_mask = mask = X[:, best_feature] <= best_threshold
    best_left_X,best_left_y = X[best_mask],y[best_mask]
    best_right_X, best_right_y = X[~best_mask],y[~best_mask]
    d = {
        "leaf" : False,
        "feature" : best_feature,
        "threshold" : best_threshold, 
        "left" : build_cart_tree(best_left_X, best_left_y, max_depth, min_samples_split, depth + 1),
        "right" : build_cart_tree(best_right_X, best_right_y, max_depth, min_samples_split, depth + 1)
    }
    # print(d)

    return d


# Test Case 1
import numpy as np
X = np.array([[1],[2],[3],[4]])
y = np.array([0,0,1,1])
print(build_cart_tree(X, y, max_depth=2, min_samples_split=2))

# Expected Output:
# {'leaf': False, 'feature': 0, 'threshold': 2.5, 'left': {'leaf': True, 'prediction': 0}, 'right': {'leaf': True, 'prediction': 1}}