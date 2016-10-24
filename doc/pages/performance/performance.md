[Back to start page](../../../ReadMe.md)

#### Notations
Some of the performance indicators used on this page depend on the following variables:
- **s** \- the size of the (square) symbols used during image approximation process
- **n** \- the total number of these available symbols
- **c** \- the count of patches to approximate (during image transformation, the image is viewed as a grid of patches)
- **uc** \- the number of *uniform* patches (they lack contrast, so approximating them is meaningless)
- **w** \- the size of the blur window (one matching aspect needs also blurred versions of the patches and symbols)

* * *

## Performance of Pic2Sym &#60;&#61; v1.3 ##

**Version 1.3** has put emphasis on *application responsiveness* and *efficiency*. The user was placed in charge of *how often to get feedback* from the application. For that:
- it implemented _charmap **dynamic** partitioning_ with a better draft generated for every new batch of symbols from the entire set. The user might adjust on the spot the size of next batches, thus the draft frequency.
- it delivers an *early preview of large charmaps* while they are loaded
- it removed any *optional nested parallelism from image approximation code from v1.2*

Outcome (compared to *v1.2*):
- **faster transformation** if the user requires no drafts, or only a few. This is because there was a lot of OpenMP code involved in the removed *optional nested parallelism*
- drafts computed based on charmap partitioning required holding a matrix of previous best matches, so this means **larger memory footprint**

* * *

### Analysis for Pic2Sym version 1.2

**Version 1.2** uses **OpenMP** for parallelism (*Visual C++ implementation of OpenMP version 2.0 from March 2002*).

The code from *version __1.1__* required some *minor rearranging and adjustments* to accomodate the *pragma clauses providing concurrency support*:
- most adaptations dealt with *loops* that previously were using *iterators* and had to switch to loop counters
- the introduced pragma clauses transformed into *code for managing eventual requests to parallelize any following code*

Outcome:
- around a ***5 \- 15% time penalty*** for *handling possible parallelization requests, even when there are none*.
	That&#39;s why *version 1.2 is slower than v1.1 on single\-core machines or when the parallelization is disabled*
- on multi\-core machines and when parallelizing transformation of consecutive rows of patches, version 1.2 needs *less than 65% image transformation time* compared to v1.1.
	Selecting other regions for parallelization or increasing the cores count would generate different durations

Below I&#39;ll present **most time\-consuming functions** while performing image transformation with *2 threads tackling consecutive rows of image patches*:<br>
![](ProfileFunctionDetails1.2_SSimWeight.jpg)<br>
Most time is spent in ***MatchEngine::assessMatch*** (98.1%). Largest chunk of the previous percent is consumed by ***StructuralSimilarity::assessMatch*** (79.5%). Next one is ***EdgeMatch::assessMatch*** (8.6%).

Therefore it&#39;s worth investigating **StructuralSimilarity**:<br>
![](ProfileFunctionDetails1.2_BlurWeight.jpg)<br>
From the 79.5% observed earlier, 47.7% is required by the block:<br>
***&#35;pragma omp parallel if(ParallelizeMp_ssimFactors)*** \- transformed into ***MatchParams::computeSsim&#36;omp&#36;4*** (**4\-th** active ***omp parallel*** clause from ***MatchParams::computeSsim***)

**Red** marks the *most expensive call*: ***GaussianBlur*** from *opencv_imgproc.dll* (27.1%).<br>
**Green** covers the remaining 18.5% spent performing ***basic matrix operations*** from *opencv_core.dll*.

So, ***GaussianBlur*** calls alone (27.1%) represent **1/4 of total transformation time** and consume more time than evaluating *all other match aspects together* (&#126;19%).

* * *

### Analysis for Pic2Sym version 1.1

The changes in **version 1.1** of Pic2Sym impact the performance as follows:
- Using the **Hybrid Result** mode incurs *additional cost*:
	- first deciding *which is a better approximation* of the patch: *a selected symbol* or *the blurred version of that patch*
	- secondly \- *combining the 2 versions* based on the *weights resulted from previous step*
- **Uniform patches** are *approximated by their blurred form*, instead of the normal transformation process from version 1.0. This extra logic generally **reduces overall processing time**, except for the case when there are very few uniform patches. It also means it&#39;s more difficult to provide accurate time-estimation formulae
- **Refactorization incurred minor penalty** for *splitting old classes*, *runtime allocations and handling of some (more / larger / polymorphic) objects*

Based on the introduced notations, **hybrid mode cost** for a *color image* is:<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(c-uc) \* (9\*s^2 + 12)<br>

Since the changes don&#39;t affect the matching aspects, the analysis for version 1.0 from below should be still mostly valid.

* * *

### Analysis for Pic2Sym version 1.0

Profiling results show that during image approximation under default settings, 80% from the time the processor is busy computing [structural similarity][] (*StructuralSimilarity::assessMatch* \-\> *MatchParams::computeSsim*):<br>
![](ProfileSummary.jpg)<br>
The 2 dll\-s from the image are **imgproc**, which provides an *image blur* function, and **core**, used for *basic matrix operations*, like addition and element\-wise multiplication.

The emphasis of next ideas will be on **imgproc**, as the next capture might suggest:<br>
![](ProfileFunctionDetails.jpg)<br>
This *Function Details* view presents the operations from **MatchParams::computeSsim** together with the percent of the total time required by each. The lines not relevant here were removed from the *Function Code View*.

The red rectangles tackle the mentioned image blur (*GaussianBlur*) from **imgproc** dll. The overall duration for the blur is 28.9% (28.8% is displayed in the top\-right red rectangle, probably because it ignores call costs).

The variables *blurredPatchApprox* (line 234) and *variancePatchApprox* (line 237) do actually require each same *GaussianBlur* computation method.
However, their actual evaluation in the implemented optimized algorithm _needs **\>8 x less time** each_.
Lines 241-257 from the code check in Debug mode that the computations are the same when using normal and optimized methods.

The call to *computeVariancePatch* on line 226 involves also 2 other sub\-calls to *GaussianBlur*, but they are performed only once per patch.
Despite the fact its single\-time workload is distributed among all the symbols compared with the given patch, *computeVariancePatch* still averages to 0.7% from the total time and it is categorized as ***Other*** (0.9%) in ***Called functions***. This is perhaps more evident in the view below:<br>
![](ProfileCallerCallee.jpg)<br>
The transformation time for the non-optimized algorithm would be around 57% longer than for the implemented version (100% + 2*28.8% = ~157%).

When disabling [Structural Similarity][], there are fewer differences among the remaining matching aspects. Smoothness and contour match aspects require a bit more time than the rest.

All matching aspects share the values that normally would be recomputed. First one who needs such a value will evaluate it; the next ones will simply use it.
The total approximation effort is split in 3:

1.	Symbol\-specific data preparation - various masks, a negative, a blurred version and others, all obtained when the glyph is loaded and resized
2.	Transformation\-specific values computation - patch area, some limits and others, all generated at the start of the transformation
3.	Actual approximation of each patch

Below I&#39;ll compute first the overall transformation complexity, ignoring (1) and (2) from above. Then I&#39;ll enumerate the distinct contribution of each matching aspect, without the part when they compute/read shared data.
For simplicity I won&#39;t count compare, saturation casts and loop index increment operations and I assume that all arithmetic operations will need the same time.
The expressions make use of the notations introduced at the top of the page.

Here are the estimations for employed OpenCv functions:
- ***countNonZero*** : 0.5\*s^2   (on average, half of the mask elements might be 0)
- ***mean*** : s^2 + 1   (with mask parameter)
- ***meanStdDev*** :  2.5\*s^2 + 3   (with mask parameter)
- ***GaussianBlur*** : 2\*w \* s^2   (for a separable kernel method; otherwise it would be (w\*s)^2 ;   For related details, see [this](http://www.mathworks.com/matlabcentral/fileexchange/28238-kernel-decomposition)\)

**Overall transformation complexity** for a *color* image:<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;c \* { n \* [s^2 \* (2\*w +29.5) + 61 ]   +   s^2 \* (4\*w+30)   +   4\*s   +   8 }.

**Separately** and **without the parts computing/reading of shared data**, each match aspect contributes as follows:
- ***[Structural Similarity][]*** : c \* { n \* [s^2 \* (2\*w+19) + 6 ]   +   s^2 \* (4\*w+3) }
- ***Fit under*** : 3\*c\*n
- ***Fit edge*** : (2.5\*s^2 + 7) \* c\*n
- ***Fit aside*** : 3\*c\*n
- ***Contrast*** : 1.5\*c\*n
- ***Gravity*** : 4\*c\*n
- ***Direction*** : 6\*c\*n
- ***Larger symbols*** : 3\*c\*n
- ***Blank threshold*** : 4\*c\*n

For **c**=27540, **n**=125, **s**=10 and **w**=11 (values used for the profiled transformation), the ratio of the complexities of *[Structural Similarity][]* and the *Overall Transformation* is _**0.78617**, which is close to **79.95%** reported by the profiler_.
The last discussed image demonstrates that the measured Structural Similarity computations concern almost entirely only Structural Similarity \- below the underlined computeVariancePatch, only 0.06% of the time is spent in shared values management (the call to *computePatchApprox*).

The **memory required by the transformation process** is ***O***((n+c)\*s^2) where **c** is typically much larger than **n**.
For the scenario mentioned above, *maximum reported memory* was less than *114 MB*.

-----------
[Back to start page](../../../ReadMe.md)

[structural similarity]:https://ece.uwaterloo.ca/~z70wang/research/ssim