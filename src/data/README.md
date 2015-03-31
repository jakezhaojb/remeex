Data usages
===

Where?
---
Located at: crunchy6: /scratch/jz1672/remeex/features

Formats
---
The feature extracted from each song is stored by `np.savetxt` as separate file. This is good for small dataset experiments essentially.

Use `np.loadtxt(path/to/file, delimiter=',')` load the file and the resulting matrix has the shape `[number_of_segments, dim_of_feature]`.

Corresponding annotations (being segmented) are stored in melody\_type[123]

Note
---
Normally it is not needed to read feature extraction code. If you have any question, just come to Jake.

Enjoy.

