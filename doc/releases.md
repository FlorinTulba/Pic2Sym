## Released Versions:

[**Version 1.2**](../../version_1.2/ReadMe.md):
- provides ***improved speed*** compared to v1.1 for **multi\-core machines**. For that, it offers several **parallelization switches** configurable in [res/varConfig.txt](../../version_1.2/res/varConfig.txt) (*no recompilation needed when modified*).
 	There is a **global UsingOMP switch**, which *disables parallelization completely when false*.<br>
    The other parallelization switches command *sequential / concurrent behavior in different regions of the application*. The only **restriction** is these areas *shouldn&#39;t be contained in parent zones that must be parallel, too*, that is *nested parallelism is disabled*.
- is a bit slower than v1.1 on single\-core machines or when the parallelism is disabled.

- - -

[**Version 1.1**](../../version_1.1/ReadMe.md):
- tackles the **problem of less\-inspired approximations**. It uses the fact that *blurring smoothens any sudden transitions*, which helps against *disturbing high\-contrast inside paches*, but also lets *neighbor patches stay more similar*.
    The solution was to **build each result patch** by *combining* the *approximation (from version 1.0)* with the *blurred version* of the initial patch. It *weighs them* based on *which resembles more to the original patch*.<br>
    For **inferior approximations**, the ***hybrid result*** should be **dominated by the blur**.<br>
    **Superior approximations** will make the **blur less visible than the symbol**.<br>
    Thus, this method provides a *direct measure of how appropriate is approximation* for each patch, while *maintaining inter\-patch cohesion*
- **skips uniform patches**, generating just a *blur result patch* for each such region

- - -

[**Version 1.0**](../../version_1.0/ReadMe.md):
- provides a set of simple operations involved in the image approximation workflow
- generates ***prettier results*** using *small, bold symbols and for images with large mild\-texture regions with clear and long contours* (like large faces, buildings and clouds).
