- After the last upgrade the model now seems to perform worse according to veriy-model, even when restoring to 200 epochs, why could that be? What changes did were made to training in the last edit apart from the epoch change?

- TPR and FPR are identical across several scenarios for some of the detectors, this seems suspicious

- ADWIN is now showing a very high false alarm rate way above the $\delta = 0.05$, the sublety is that ADWIN's any-time valid false alarm guarantee is different from PITMonitor's, it's something like a bound on the amount of streams that will false alarm at least once, not the FPR itself I think so setting delta=0.05 gives an unfair comparison, have to think of a better way of doing this.

- More generally, I feel the parameter defaults might be setting up many of the detectors to perform unfairly bad, these should be set according to the right methodology for each if that's necessary. I dont know...

- Fix this error: Font 'default' does not have a glyph for '\u2212' [U+2212], substituting with a dummy symbol.