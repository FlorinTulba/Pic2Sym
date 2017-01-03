## Testing Pic2Sym v2.0 (*64\-bit Windows only*) ##

[Back to start page](../../../ReadMe.md)

It&#39;s *recommended* to read first the ***Conclusions*** the ***[examples page][examples]***, to see *what to expect* and understand *what images and font types are more suitable for the transformation*.

Font families with a moderate number of symbols let Pic2Sym produce its results faster. In any case, as soon as the result appears satisfactory-enough, or when the estimated remaining transformation time is not acceptable, the user can cancel the process while provided with a saved version of the last generated draft result.

The **optional** *free* font ***BpMono bold*** with less than 200 symbols can be installed (with *administrator rights*) from [here][BpMonoBold] or from **[res/][ResFolder]** folder in order to be visible while running the application. *Installing* fonts requires just double-clicking the font file and then pressing **Install** button. *Uninstalling* steps: **Start** button; click **Run**; type &quot;**%windir%\\fonts**&quot;; Right click on desired font; Choose **Delete**.

Now the steps for testing Pic2Sym (**no administrative privileges required**):

<table style="width:100%; margin-left:0; margin-right:0" border="0" cellpadding="0" cellspacing="5">
	<tr valign="top" style="vertical-align:top">
		<td width="60%" align="justify" style="text-align:justify; padding-left:0; padding-right:0">
			1. Double\-click **[bin][BinFolder]/Pic2Sym.msi** to install the application.
			<p>
			The installer:
			<ul>
				<li>allows choosing the directory for the installation</li>
				<li>can create Shortcut, Quick-launch and Start Menu items</li>
				<li>will create an entry within installed programs which looks like the image from the right</li>
			</ul>
		</td>
		<td>
			![](installedApp.jpg)
		</td>
	</tr>
	<tr valign="top" style="vertical-align:top">
		<td align="justify" style="text-align:justify; padding-left:0; padding-right:0">
			2. Launch **Pic2Sym.exe**, then activate the [*Control Panel*][CtrlPanel] from the *main window* by:
			<ul>
				<li>either pressing **Ctrl\+P**</li>
				<li>or by clicking the tool marked with red within the toolbar</li>
			</ul>
		</td>
		<td>
			![](mainWindow.jpg)
		</td>
	</tr>
	<tr valign="top" style="vertical-align:top">
		<td align="justify" style="text-align:justify; padding-left:0; padding-right:0">
			3. Adjust the settings for a *first* ***rapid*** transformation based on the highlighted controls from the right:
			<ul>
				<li>select a **small image** to transform</li>
				<li>select a *font family* with **less than 400 symbols**. The *symbols window* should provide information about the size of the loaded font family:<br>
    ![](smallFontFamily.jpg)</li>
				<li>bring **[Structural Similarity][] slider to 0**</li>
				<li>*optionally* set **Batch syms** slider on a **small non-zero value** if you prefer *more frequent draft results*</li>
			</ul>
		</td>
		<td>
			![](rapidTestConfig.jpg)
		</td>
	</tr>
	<tr valign="top" style="vertical-align:top">
		<td colspan="2" align="justify" style="text-align:justify; padding-left:0; padding-right:0">
			4. Finally hit **Transform the Image**. The approximation process can be **canceled** at any time by **pressing ESC**. At the end of the transformation you&#39;ll be able to *inspect the result* with the *Transparency slider* and the *Zoom feature* from the main window
		</td>
	</tr>
</table>

--------
[Back to start page](../../../ReadMe.md)

[ResFolder]:../../../res/
[BinFolder]:../../../bin/
[BpMonoBold]:http://www.dafont.com/bpmono.font
[examples]:../results/results.md#Conclusions
[CtrlPanel]:../CtrlPanel/CtrlPanel.md
[msvcp120.dll]:http://files.dllworld.org/msvcp120.dll-12.0.21005.1-64bit_3075.zip
[msvcr120.dll]:http://files.dllworld.org/msvcr120.dll-12.0.21005.1-64bit_3122.zip
[vcomp120.dll]:http://down-dll.com/index.php?file-download=vcomp120.dll&arch=64bit&version=12.0.21005.1&dsc=Microsoft%AE-C/C++-OpenMP-Runtime#
[comdlg32.dll]:http://files.dllworld.org/comdlg32.dll-6.1.7601.17514-64bit_181.zip
[advapi32.dll]:http://files.dllworld.org/advapi32.dll-6.3.9600.17031-64bit.zip
[Structural Similarity]:https://ece.uwaterloo.ca/~z70wang/research/ssim