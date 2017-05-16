## Released Versions:

[**Pic2Sym with several CUDA prototype algorithms**](../../prototypesCUDA/ReadMe.md):

- is built on top of version 2.0, but uses *single-precision floating point values*, which are available on all GPU-s. Version 2.0 is using double-precision
- offers 2 versions of the **Box blur** algorithm:
	- one executing the blur *entirely on the GPU*
	- the other one using the GPU only for the *initialization of the rolling sums* of next iteration, while the CPU finishes the previous iteration
- adapted and improved an existing CUDA version of the **Stack blur**


* * *

[**Version 2.0**](../../version_2.0/ReadMe.md):

- provides ***shorter image transformation time*** and ***enhanced quality of the result*** by **pruning the glyphs** and then **regrouping them by similarity**. Several methods of *filtering* and *clustering* are available
- isolates ***reusable data related to the symbol sets*** (*clustering information* and some details for the *preselection process* described below) and **keeps it on disk** (compressed or not) to be just reloaded when needed again
- introduces a ***symbols preselection process*** (which is particularly advantageous for *larger font sizes*) for the glyphs competing during an image transformation:
	- comparing *initially* **tiny versions** of the glyphs from the symbol set and of the patches from the image
	- a *second step* tackles normal-size glyphs and patches and just **selects the best matching symbol among the few good candidates** resulted from the *first pass*
- *reorders and evaluates user&#39;s requested matching aspects* to allow **detecting as cheap and as early as possible** when a symbol *cannot be the best match for a given patch of the image*
- offers several *image blurring techniques*, as *alternatives* for the **Gaussian blur** (up to this point ***the most expensive operation performed during the image transformation*** when the *Structural Similarity* matching aspect is enabled)
- allows multiple *non\-conflicting* **user commands** running in **parallel**

* * *

[**Version 1.3**](../../version_1.3/ReadMe.md):

- allows **controlling** ***application&#39;s responsiveness***:
	- the *approximation process* can be **canceled with ESC**, while the *available generated draft gets saved*
	- the user can **dynamically select the size of next batches** of symbols used to generate new better drafts, thus setting *draft update rate*
	- loading *large symbol sets* delivers an **early preview of those new glyphs**
- uses **fewer** [**switches for parallelism**](../../version_1.3/res/varConfig.txt) concerning the *image transformation process*. However, it&#39;s **faster than v1.2**, due to its obtained simplicity, as long as only a *few drafts* are to be generated.

- - -

[**Version 1.2**](../../version_1.2/ReadMe.md):

- provides ***improved speed*** for **multi\-core machines** compared to v1.1. For that, it offers several **switches for parallelism** configurable in [res/varConfig.txt](../../version_1.2/res/varConfig.txt) (*no recompilation needed when modified*).
 	There is a **global UsingOMP switch**, which *disables parallelism completely when false*.<br>
    The other switches for parallelism command *sequential / concurrent behavior in different regions of the application*. The only **restriction** is these areas *shouldn&#39;t be contained in parent zones that must be parallel, too*, that is *nested parallelism is disabled*.
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
