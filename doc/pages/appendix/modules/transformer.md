## Image Transformer module

[Back to the Appendix](../appendix.md) or jump to the [start page](../../../../ReadMe.md)

-------

![](TransformerRelated_classes.jpg)<br>

When an image transformation is started, ***Transformer***:

- starts a ***Timer***, configures the notifier ***PicTransformProgressNotifier*** and the actions ***TimerActions_ImgTransform*** for this timer

- ensures the image ***Img*** is resized appropriately (***ResizedImg***) and checks if it&#39;s not processed already under current settings with ***ResultFileManager*** (This class also tackles saving the results for completed / canceled transformations)

- initializes the values of the first draft result or just resets them when reprocessing same image under different settings. When **symbols preselection** is enabled, it applies the same initialization considering the tiny image and glyphs

- improves the draft with each new symbols batch (provided as a pair of indices within the symbol set)

- saves a ***TransformTrace*** (in Debug mode, if the transformation wasn&#39;t canceled) that displays several ***MatchParams*** fields and a matching score for every approximated ***Patch*** by a ***BestMatch***

When **symbols preselection** is enabled, a template of the short candidate list (***TopCandidateMatches***) has to be provided to the draft improver (***MatchEngine***), who will fill it with a few most-promising tiny symbols candidates from the batch. When the batch contains no acceptable candidates, the short list remains empty. After this preselection step, the ***MatchEngine*** determines the best candidate (normal size, not tiny) from the short list.<br>

In the case of disabled **symbols preselection**, the ***MatchEngine*** determines the best candidate (normal size) directly from the whole batch.<br>

If the best found symbol among the batch improves the existing draft, the draft gets updated (***ApproxVariant*** within a ***BestMatch*** object).<br>

Every stage (task) of the approximation process (job) is monitored by a separate ***TaskMonitor*** and an unique ***JobMonitor***.<br>

-------

[Back to the Appendix](../appendix.md) or jump to the [start page](../../../../ReadMe.md)
