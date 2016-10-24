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
- lets the user select *which critical paths to parallelize* on *multi\-core machines*
- allows multiple *non\-conflicting user commands* running in *parallel*
- tackles well transformations based on font families with *less than 400 symbols*.

Further speed improvements could be obtained by:
- involving *other processing devices* from one&#39;s machine, like GPU\-s, especially in areas like the *Gaussian blur*
- saving *patch statistics about an image* for every *patch-size* requested, to shorten its retransformation under other conditions
- starting a *machine learning background thread* that learns while application is idle and shares its gathered experience when an actual image transformation is requested

-----
[Back to start page](../../../ReadMe.md)

