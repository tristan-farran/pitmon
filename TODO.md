
- TPR and FPR are identical across several scenarios for some of the detectors, this seems suspicious - investigate

- ADWIN is now showing a very high false alarm rate way above the $\delta = 0.05$, the sublety is that ADWIN's any-time valid false alarm guarantee is different from PITMonitor's, it's something like a bound on the amount of streams that will false alarm at least once, not the FPR itself I think so setting delta=0.05 gives an unfair comparison, think of a better way of doing this and implement it in the code and paper

- More generally, I feel the parameter defaults might be setting up many of the detectors to perform unfairly bad, these should be set according to the right methodology for each if that's necessary. Address this

- Fix this error: Font 'default' does not have a glyph for '\u2212' [U+2212], substituting with a dummy symbol.

- Improve plot readability, eyeball what we've got now and see what's wrong with it, some examples: some spacing is suboptimal, the single runs look bad and are never shown in the paper (one should be), the delay distributions look good but aren't in the paper (they should be), PITMonitor is shown now in fig_delay_distributions.png, where it doesn't detect at all - which is correct except its shown hugging the y-axis rather than in its usual spot some space to the right as in the plots where it does detect. etc.