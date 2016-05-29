## Conclusions and further directions ##
[Back to start page](../ReadMe.md)

By now (v1.3), the application:
- offers *various match aspects* for deciding which symbol will approximate an image patch
- provides either *raw image approximations*, or a satisfactory *hybrid result* for more complex content
- lets the user select *which critical paths to parallelize* on *multi\-core machines*
- allows *transformations* to be *canceled*, while saving the available draft result
- presents several *coarser drafts* while *refining the result* and lets the user decide their frequency
- displays *early previews* of *large font families* while they get loaded
- tackles well transformations based on font families with *less than 400 symbols*.

Using *larger symbol* sets would be possible only by *removing some undesired glyphs* and *clustering* the rest. *Patches* could be *similarly clustered*, at least for *draft results*.

Further speed improvements could be obtained by:
- reducing the count of *expensive match assessments*:
	- comparing a *smaller version* of the patch with same\-size glyphs. All such small symbols that don&#39;t resemble the mini\-patch can be ignored. Only the rest need to perform the compare for full\-size versions
	- skip computing further match aspects that *can&#39;t improve the overall match score beyond the value for the best match known at that moment*
- involving *other processing devices* from one&#39;s machine, like GPU\-s
- *preprocessing the symbols* from the font to be used
- maintaining *patch statistics about an image*, to shorten its retransformation under other conditions
- *saving expensive data* that is *likely to be needed again* in related future transformations
- starting a *machine learning background thread* that learns while application is idle and shares its gathered experience when an actual image transformation is requested

-----
[Back to start page](../ReadMe.md)

