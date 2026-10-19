<!--
Help text shown in the notebook form (click the "ℹ Help" toggle under each section heading).
Edit freely - no code changes needed. Rules:
* Each "## key" heading starts a help entry. Keys: section_1, section_2, section_3, section_4.
* Body is standard Markdown (use ``` fenced blocks for folder trees). Remove an entry to hide its help.
* Re-run the form cell after editing.
-->

## section_1

The script needs to know where to find your camtrapdp exports. The expected folder structure (relative to the repository root) is:

```
├── Input_Data/
│   ├── Test_Set_1/
│   │   ├── CamtrapDP_Export_Model_1/
│   │   └── CamtrapDP_Export_Model_2/
│   └── Test_Set_2/
│       ├── CamtrapDP_Export_Model_1/
│       └── CamtrapDP_Export_Model_2/
```

## section_2

Including descriptive details gives context to the report and aids interpretation. The validity of the results presented in the report are largely determined by the quality of the source data used. Provide as much detail as possible about the source of the data and how it was collated for testing.

## section_3

Recall, Precision, and F1 score can be reported for different confidence thresholds. This information can be used to identify the optimal confidence threshold (highest F1 score) for each species. For a highly detailed report, select a granular interval of confidence thresholds. For a simpler report, select a coarser interval or a single value such as >0.5. If unsure, go with the default value.

Exports from WildObs can include a combination of observations grouped by event and observations associated with individual media files. It is recommended to use event-based observations.

## section_4

The 'Full' report template contains detailed descriptive information and definition of terms that helps the reader to understand and interpret the results. This is recommended for most use cases. The 'Minimal' template is suitable for an advanced audience with a high level of technical literacy and understanding of terminology such as Recall, Precision, and F1 Score.
