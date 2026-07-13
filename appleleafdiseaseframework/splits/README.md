# Leakage-free split definitions

Place here the explicit per-partition image lists for the de-duplicated apple set,
one path per line:

    train.txt   val.txt   test.txt

These lists guarantee a leaf-level disjoint split. The training pipeline reads them
directly, and a zero image-path overlap between the three files is asserted at load time.
