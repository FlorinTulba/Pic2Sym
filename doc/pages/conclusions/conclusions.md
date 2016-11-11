## Conclusions and further directions ##
[Back to start page](../../../ReadMe.md)

By now (v2.0), the application:

- presents several *coarser drafts* while *refining the result* and lets the user decide their frequency dynamically
- displays *progress* information, *elapsed and estimated remaining time* for an image transformation or for the loading of a symbol set
- allows *transformations* to be *canceled*, while saving the available draft result
- displays *early previews* of *large font families* while they get loaded
- *preprocesses the symbols* from the font to be used:
    - can *remove several categories of undesirable glyphs*
    - offers 2 *clustering* algorithms for grouping the similar symbols. Generated clusters are saved on disk to be reused
- provides either *raw approximations of the image*, or a satisfactory *hybrid result* for more complex content
- offers *various match aspects* for deciding which symbol will approximate an image patch
- reduces the count of *expensive match assessments*:
	- can be configured to compare a *smaller version* of the patch with same\-size glyphs. Only the *tiny symbols most resembling to the mini-patch* need to perform the compare for full\-size versions. Data for all tiny symbols is saved on disk to be reused
	- *reorders and evaluates enabled  matching aspects* to allow *detecting as cheap and as early as possible* when a symbol *cannot be the best match for a given patch of the image*
- reuses *image patches information* (without saving it) between *consecutive transformations of the same image with symbols of the same size*
- saves (compressed versions of) the cluster and tiny symbols data
- offers several *image blurring techniques*, as *alternatives* to the *Gaussian blur* (up to this point *the most expensive operation performed during the image transformation* when the *Structural Similarity* matching aspect is enabled)
- lets the user select *which critical paths to be performed in parallel* on *multi\-core machines*
- allows multiple *non\-conflicting user commands* running in *parallel*
- tackles well transformations based on font families with *less than 400 symbols*.

Further speed improvements could be obtained by:

- involving *accelerator devices* from one&#39;s machine, like GPU\-s, especially in areas like the *Gaussian blur*
- dynamically disabling any features less-suited for a given scenario - like clustering for low values of the average cluster size
- rearranging data for augmenting the effects of various heuristics. For instance, when the *Larger Symbols* Matching Aspect is enabled, it will be always the first one evaluated when comparing a patch with a symbol; thus, rearranging or just iterating the symbols in reverse order of their *density* (the amount of area they consume from their square) will potentiate the effect of the *Skip Matching Aspects* heuristic, by statistically increasing the chance of finding **earlier** large-enough good matches that force lots of smaller symbols to stop competing right after their *density* evaluation

-----
[Back to start page](../../../ReadMe.md)

