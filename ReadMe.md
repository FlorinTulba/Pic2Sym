# Pic2Sym v2.0 #

**Topics:**

1. Description of Pic2Sym application \(*see below*\)
1. [Related work](doc/pages/relatedWork/relatedWork.md)
1. [Example results](doc/pages/results/results.md)
1. [Performance](doc/pages/performance/performance.md)
1. [Conclusions and further directions](doc/pages/conclusions/conclusions.md)
1. [Testing the provided binary](doc/pages/testBinary/testBinary.md) \(*64\-bit Windows only*\)
1. [Other released versions](../version_1.0/doc/releases.md)
1. [Appendix][Appendix] \(*various technical details*\)
1. [Known issues][Issues]

## Description of the application ##

*Pic2Sym* **approximates images** by a **grid of colored symbols with colored backgrounds**.

It runs on **64\-bit Windows \(at least version 7\)**. It comes with a *graphical interface* and a *console* tracing relevant events.

The symbols used for approximation are **scalable**, *preferably* **fixed\-width** fonts, like *Consolas*, *Courier New* and *Lucida Console*. [Here](https://en.wikipedia.org/wiki/Samples_of_monospaced_typefaces) are more examples of such fonts. Scalable fonts allow using virtually any size of the font while preserving their quality. Fixed\-width fonts are more evenly width\-distributed, thus more helpful to approximate random patches.

The **result** of the image transformation is a grid of __*square* cells__. All the symbols are **preprocessed** to fit such cells, so the *original symbol set* is *altered*. Therefore the **output is simply a new image**, and cannot be saved neither as a character table in HTML, nor as a text file (like similar applications do). Displaying the output into a console is also ruled out, first because there are only a couple of console font sizes available and secondly, because consoles usually provide only 16 colors.

Several **features** from *Pic2Sym* (explained in more detail in the [Appendix][Appendix], [Control Panel][CtrlPanel] and in the [*configuration*](res/varConfig.txt) file):
- Transformations display **intermediary results (drafts)** *as often as adjusted during the ongoing process* \[see (*1*) from the image below \- &#39;*Batch Symbols*&#39; explained in [Control Panel][CtrlPanel]\]. When **aborting** a transformation (with *ESC* key), the *last draft gets saved*
- The *artistic quality* of generated results can be improved by:
	- making sure that **_poorly_ approximated image patches will remain discreet** \[see (*2*) from image below - &#39;*Hybrid Result*&#39; explained in [Control Panel][CtrlPanel]\]
	- **removing undesirable glyphs** from the symbol set, like *barely readable ones*, or *large rectangles* a.o. (This *reduces transformation time*, since it shrinks the symbol set)
- The symbol set is **arranged by similarity of the glyphs**, resulting a **set of clusters** (which gets *saved* and can be *reused afterwards*). This **accelerates the image transformation** because of *less compare operations* between patches and symbols:
	- image patches are **first** compared just with a ***representative of each such cluster***
	- **if the cluster representative is similar-enough to the patch**, then *all individual symbols from the group compete* to determine the best match for the patch among them
	- but **otherwise, no symbol from that group needs to be compared against the given patch**
- During an image transformation, the *competing glyphs* can also undergo a ***symbols preselection process*** (which is particularly advantageous for *larger font sizes*):
	- comparing *initially* **tiny versions** of the glyphs from the symbol set and of the patches from the image
	- a *second step* tackles normal-size glyphs and patches and just **selects the best matching symbol among the few good candidates** resulted from the *first pass*
- The user might **specify several aspects of interest** for every transformation (like a preference for results with many *large symbols* \[(*3*) from image below\], or if the patches might get approximated by less similar symbols, but with *better contrast* \[(*4*) from image below\]). See [Control Panel][CtrlPanel] for the entire list of *matching aspects*
- All requested matching aspects (mentioned above) will use a *heuristic evaluation* and will get rearranged in a *particular order* that allows **detecting as cheap and as early as possible** when a symbol *cannot be the best match for a given patch of the image*. A **surprising consequence**:
	- when using only a single *complex* enabled matching aspect, this must be evaluated for each pair symbol\-patch
	- but when using the *same complex matching aspect* *together with a few simpler enabled aspects*, this allows *skipping often enough the evaluation of the most complex ones*. In turn, this means **a faster transformation, despite there are more enabled matching aspects to consider** compared to the first case
- The application is **faster on multi-core machines** (unless it&#39;s configured for no parallelism)
- **Elapsed and estimated remaining time** for *image conversions* and also for *loading symbol sets* are provided *within the console window*
- Multiple *non\-conflicting* **user commands** can run in **parallel** (e.g. clicking &#39;*Instructions*&#39; \[(*5*) from image below\] while both, a *large symbol set* \[(*6*) from image below\] and also a *large image* \[(*7*) from image below\] are loading)

Next comes a snapshot of the application *while processing an image*:<br>
![](doc/pages/DuringTransformation.jpg)

- - -

The **main window** allows **comparing** the *original image* with the *approximation result* using the **Transparency slider** at the bottom of the window.<br>
**Magnifying a region** is possible using the toolbar or the mouse wheel. \(Large zoom factors display even the values of the pixels.\)<br>
![](doc/pages/MainAfterTransform.jpg)<br>
Same window is used to [present mismatches and wrongfully filtered symbols detected during Unit Tests](doc/pages/UnitTesting/UnitTesting.md).

- - -

**Symbol set window** displays the **symbols used for the approximation**, one page at a time. Its status bar provides *complete information about these symbols* \(font family, style, size, encoding and total count\).<br>
![](doc/pages/CmapViewer.jpg)<br>
The presented glyphs are *already resized to fit in square cells* (using *widening* and sometimes additional *minor translation / crop* operations).<br>*Blanks and exact\-duplicates were removed* and the remaining symbols were ***reordered*** **primarily by similarity** (*showing larger groups first*) and **finally by their &#39;density&#39;** (*how much surface from their patch they consume*).<br>The glyphs with **inverted colors** were detected as **undesirable symbols** (they just induce a *slower and lower quality transformation*). *Any selected* category of *undesired symbols* can be **easily hidden** from a [*configuration file*](res/varConfig.txt) to *improve the conversion process*.

- - -

The **Control Panel** contains all the necessary controls to **customize and generate approximations for various images**:<br>
![](doc/pages/CtrlPanelAndInstructions.jpg)<br>
The user can **display / hide** it using **Ctrl\+P** or the **last tool from the toolbar** from any other window:<br>
![](doc/pages/CtrlPanelWithinToolbar.jpg)<br>
[Here][CtrlPanel] are more explanations about the controls.
_ _ _

To **leave the application**, please **activate a window** (except the console and the Control Panel) and **press ESC** \(check these [limitations][Issues] on that\).

* * *

Kindly address any observations, suggestions or questions to me using ***<florintulba@yahoo.com>***.<br>&copy; 2016 Florin Tulba (GNU AGPL v3 license)

* * *

**Note:**
*For exemplifying the conversions performed by the project, I&#39;ve applied the transformations on several images provided by 3 friends with well\-developed artistic abilities, all of us sharing the love for nature. So, many thanks to [Trifa Carina](https://www.facebook.com/trifa.carina), [Muntean Flavia](https://www.facebook.com/darkfavy) and [Rosca Adrian](https://www.facebook.com/rosca.adrian.9). If you need those beautiful pictures or similar ones, please contact them.*


[CtrlPanel]:doc/pages/CtrlPanel/CtrlPanel.md
[Appendix]:doc/pages/appendix/appendix.md
[Issues]:doc/pages/issues/issues.md
