## Conclusions and further directions ##
[Back to start page](../ReadMe.md)

This first release provides a set of simple operations involved in the image approximation workflow.
It generates *prettier results* using *small, bold symbols and for images with large mild\-texture regions with clear and long contours* (like large faces, buildings and clouds).

*Next versions* should focus on *faster transformation* and *more meaningful results*. Image and symbols preprocessing, parallelization of critical paths might be worth investigating.

For more meaningful results, some undesired symbols could be removed.
Another track would be to avoid generating large regions where consecutive patches are largely\-distinct and besides, each patch is also a symbol with high\-contrast. Such regions don&#39;t look great and invite viewer&#39;s attention, too.

Using larger symbol sets would be possible only by clustering them, or performing the transformation using GPUs power and parallelization.
Images could also collect some patch statistics at the start of transformation and thus direct better the selection of appropriate symbols.

-----
[Back to start page](../ReadMe.md)

