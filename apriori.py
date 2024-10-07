from itertools import combinations

# Sample transactions
transactions = [
    ['bread', 'milk'],
    ['bread', 'diaper', 'beer', 'egg'],
    ['milk', 'diaper', 'beer', 'coke'],
    ['bread', 'milk', 'diaper', 'beer'],
    ['bread', 'milk', 'diaper', 'coke']
]

# Helper function to generate all item combinations
def get_combinations(itemset, length):
    return [list(combination) for combination in combinations(itemset, length)]

# Get frequent itemsets that meet the minimum support threshold
def get_frequent_itemsets(transactions, itemsets, min_support):
    itemset_counts = {}
    for itemset in itemsets:
        count = 0
        for transaction in transactions:
            if set(itemset).issubset(set(transaction)):
                count += 1
        itemset_counts[tuple(itemset)] = count

    # Filter itemsets by minimum support
    return {itemset: count for itemset, count in itemset_counts.items() if count >= min_support}

# Apriori algorithm
def apriori(transactions, min_support):
    # Get all unique items in the transactions
    items = sorted(set(item for transaction in transactions for item in transaction))

    # Find frequent 1-itemsets
    current_itemsets = [[item] for item in items]
    frequent_itemsets = {}
    k = 1

    while current_itemsets:
        # Get frequent itemsets of length k
        frequent_k_itemsets = get_frequent_itemsets(transactions, current_itemsets, min_support)
        frequent_itemsets.update(frequent_k_itemsets)

        # Generate new itemsets of length k + 1
        k += 1
        current_itemsets = get_combinations(items, k)
        # Remove itemsets that do not contain any frequent subsets
        current_itemsets = [
            itemset for itemset in current_itemsets
            if all(tuple(subset) in frequent_itemsets for subset in combinations(itemset, k - 1))
        ]

    return frequent_itemsets

# Minimum support threshold
min_support = 2

# Run Apriori
frequent_itemsets = apriori(transactions, min_support)

# Output the results
print("Frequent Itemsets:")
for itemset, count in frequent_itemsets.items():
    print(f"{list(itemset)}: {count}")
