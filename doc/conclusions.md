## Conclusions and further directions ##
[Back to start page](../ReadMe.md)

By now, the application:
- offers *various match aspects* for deciding which symbol will approximate an image patch
- provides either *raw image approximations*, or a satisfactory *hybrid result* for more complex content
- lets the user select *which critical paths to parallelize* on *multi\-core machines*
- tackles well font families with *less than 400 symbols*.

A more pleasant user experience could come from:
- allowing *tasks* to be *canceled*
- presenting several *coarser drafts* while *refining the result*
- *prioritizing jobs* to deliver first the *rapid feedback expected by user* and continue with remaining computations afterwards
- involving *other processing devices* from one&#39;s machine, like GPU\-s
- *preprocessing the symbols* from the font to be used
- maintaining *patch statistics about an image*, to shorten its retransformation under other conditions
- *saving expensive data* that is *likely to be needed again* in related future transformations
- starting a *machine learning background thread* that learns while application is idle and shares its gathered experience when an actual image transformation is requested

Using *larger symbol* sets would be possible only by *removing some undesired glyphs* and *clustering* the rest. *Patches* could be *similarly clustered*, at least for *draft results*.

* * *

### Released Versions:

[**Version 1.2**](../../version_1.2/ReadMe.md):
- provides ***improved speed*** compared to v1.1 for **multi\-core machines**. For that, it offers several **parallelization switches** configurable in [res/varConfig.txt](../res/varConfig.txt) (*no recompilation needed when modified*).
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

-----
[Back to start page](../ReadMe.md)

