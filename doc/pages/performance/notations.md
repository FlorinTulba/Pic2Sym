Back to the [Performance](performance.md) topics or jump to the [Main](../../../ReadMe.md) page

### Notations

**Some of the performance indicators relevant to the image transformation depend on the following variables:**

- **d** - the dimension of the symbols used during image approximation process
- **td** - tiny symbols dimension (symbols clustering and preselection use smaller symbols, with size *td* instead of *d*)
- **s** - the total number of the available symbols within a given font family
- **fs** - the number of symbols filtered (removed) by the filters applied on the entire set
- **acs** - average cluster size for the groups that were able to form from the unfiltered symbols (*s-fs*)
- **sll** - short list length used by the symbols preselection mechanism
- **p** - the count of patches to approximate (during image transformation, the image is viewed as a grid of patches)
- **up** - the number of *uniform* patches (they lack contrast, so approximating them is meaningless)
- **w** - the size of the Gaussian-blur window (one matching aspect needs also blurred versions of the patches and symbols)

-----------

Back to the [Performance](performance.md) topics or jump to the [Main](../../../ReadMe.md) page
