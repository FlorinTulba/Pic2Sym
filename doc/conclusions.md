## Conclusions and further directions ##
[Back to start page](../ReadMe.md)

Features introduced in **Version 1.1**:
- tackle the **problem of less\-inspired approximations**. It uses the fact that *blurring smoothens any sudden transitions*, which helps against *disturbing high\-contrast inside paches*, but also lets *neighbor patches stay more similar*.
    The solution was to **build each result patch** by *combining* the *approximation (from version 1.0)* with the *blurred version* of the initial patch. It *weighs them* based on *which resembles more to the original patch*.<br>
    For **inferior approximations**, the ***hybrid result*** should be **dominated by the blur**.<br>
    **Superior approximations** will make the **blur less visible than the symbol**.<br>
    Thus, this method provides a *direct measure of how appropriate is approximation* for each patch, while *maintaining inter\-patch cohesion*
- **skip uniform patches**, generating just a *blur result patch* for each such region

- - -

*Next versions* should focus on *faster transformation*. Image and symbols preprocessing, parallelization of critical paths might be worth investigating.

Using larger symbol sets would be possible only by removing some undesired glyphs and clustering the rest, or performing the transformation using GPUs power and parallelization.

Images could also collect some patch statistics at the start of transformation and thus direct better the selection of appropriate symbols.

* * *

### Previous Versions:

**Version 1.0** provided a set of simple operations involved in the image approximation workflow.
It generated *prettier results* using *small, bold symbols and for images with large mild\-texture regions with clear and long contours* (like large faces, buildings and clouds).

-----
[Back to start page](../ReadMe.md)

