for a in range(10):
    print("a:", a)
    for b in range(20):
        print("b:", b)
        if a==5:
            # Break the inner loop...
            break
    else:
        # Continue if the inner loop wasn't broken.
        continue
    # Inner loop was broken, break the outer.
    break