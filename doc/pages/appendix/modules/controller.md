## The Controller module

[Back to the Appendix](../appendix.md) or jump to the [start page](../../../../ReadMe.md)

-------
![](ControllerRelated_classes.jpg)<br>
The ***Controller*** manages the other modules through ***IController*** and all the additional interfaces provided by it:

- _getControlPanelActions()_ provides ***IControlPanelActions*** \- methods to address each action from [**Control Panel**][CtrlPanel], plus a method for invalidating font types that cannot be processed
- _getPresentCmap()_ provides ***IPresentCmap*** \- support for *displaying a page of glyphs from current charmap* (see ***(I)CmapInspect***, ***(I)CmapPerspective***, ***(I)UpdateSymsAction*** and ***LockFreeQueue***)
- _getGlyphsProgressTracker()_ provides ***IGlyphsProgressTracker*** \- *timing for loading and preprocessing* of a new / updated set of glyphs (see ***Timer***, ***TimerActions (Glyphs Update)*** and ***SymsUpdateProgressNotifier***)
- _getPicTransformProgressTracker()_ provides ***IPicTransformProgressTracker*** \- tracking the *progress during the picture approximation* process (see ***Timer***, ***TimerActions (Image Transform)*** and ***PicTransformProgressNotifier***); it also signals when a transformation couldn&#39;t start
- _getUpdateSymSettings()_ provides ***IUpdateSymSettings*** \- updates the symbol settings ***(I)SymSettings*** with some new valid values

Several Unit Tests need *creating custom lists of symbols* from the charmaps of various fonts. The ***Controller*** tackles this job through ***(I)SelectSymbols***.<br>

The ***Controller*** is responsible also for prompting the user with ***SettingsSelector*** for a settings file to be loaded or saved.<br>

It uses a job monitor for loading glyphs and another one for transforming images (see ***AbsJobMonitor***). The progress of these jobs is reported through ***IProgressNotifier***.<br>

The results from the image transformations can be evaluated within the ***Comparator*** window.<br>

While loading a new symbol set, some of the initial effort is directed towards displaying a first page of symbols. This was implemented using an additional thread, while the initial thread updates the GUI by consuming any enlisted ***IUpdateSymsAction*** from a ***LockFreeQueue***. Possible action types:

- updating the status bar from the symbol set window to present the total count of glyphs from the set
- reporting current progress of the loading job
- displaying the first page of the set

The ***Patch***-es to be transformed are of 2 types: normal-size or tiny versions. The tiny ones are used when the *preselection mechanism* is enabled.

Finally, the ***Controller*** orchestrates the interaction between ***(I)MatchEngine***, ***(I)FontEngine***, ***(I)Transformer***, ***Img***, ***(I)Settings(RW)*** and ***(I)ResizedImg***.

-------
[Back to the Appendix](../appendix.md) or jump to the [start page](../../../../ReadMe.md)

[CtrlPanel]:../../CtrlPanel/CtrlPanel.md
