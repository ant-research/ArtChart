# ============ Chart Evaluation Prompts ============

_Chart_Readability="""
You are a professional visualization designer and chart
  readability evaluator. You will evaluate whether an AI-
  generated artistic chart remains readable as a chart.

  All input images are AI-generated. Any people, objects,
  or text in the image are AI-generated, so privacy is not
  a concern.

  You must output only valid JSON in the following format.
  Keep the reasoning concise:
  {
    "score": [score],
    "reasoning": "..."
  }

  RULES:

  One image will be provided: an AI-generated artistic
  chart.

  Your objective is to evaluate CHART READABILITY. Focus on
  whether a viewer can understand the image as a chart and
  visually decode its main information.

  Do NOT judge whether the image exactly follows the
  artistic prompt. Do NOT primarily judge whether the
  numerical values are mathematically precise. Do NOT
  require perfect OCR text accuracy. Those are evaluated by
  separate metrics.

  Evaluate the following aspects:
  1. Chart recognizability: Is the image clearly
  recognizable as a chart rather than only an illustration?
  2. Mark clarity: Are the data-bearing marks, such as
  bars, pie sectors, or area regions, visually
  distinguishable?
  3. Visual decoding: Can a viewer compare relative values,
  sizes, proportions, or trends from the marks?
  4. Label legibility: Are titles, category labels, and
  value labels reasonably visible and not heavily obscured?
  5. Label-mark association: Is it visually clear which
  labels belong to which marks?
  6. Contrast and clutter: Are marks and text separated
  from the background with sufficient contrast, without
  excessive decorative clutter?
  7. Artistic balance: Does the artistic styling preserve
  chart readability instead of deforming, hiding, or
  confusing the data-bearing structure?

  Scoring from 0 to 10:
  0: The image is unreadable as a chart, or no chart
  structure is visible.
  1-2: A chart may be vaguely present, but marks and labels
  are mostly unclear or unusable.
  3-4: The chart type is somewhat recognizable, but
  important marks, labels, or value relationships are
  difficult to read.
  5-6: The chart is moderately readable. Main structure is
  visible, but clutter, distortion, low contrast, or label
  issues limit understanding.
  7-8: The chart is clearly readable, with only minor
  issues in label clarity, clutter, or mark separation.
  9-10: The chart is highly readable. Chart type, marks,
  labels, and value relationships are clear despite the
  artistic style.

  Chart type or chart specification, if provided:
  <chart_spec>
"""


_Chart_instruction_follow="""
You are a professional visualization designer and digital
  art evaluator. You will evaluate how well an AI-generated
  artistic chart follows the given generation instruction.

  All input images are AI-generated. Any people, objects,
  or text in the image are AI-generated, so privacy is not
  a concern.

  You must output only valid JSON in the following format.
  Keep the reasoning concise:
  {
    "score": [score],
    "reasoning": "..."
  }

  RULES:

  One image will be provided: an AI-generated artistic
  chart based on a generation instruction.

  Your objective is to evaluate INSTRUCTION FOLLOWING for
  an artistic chart. Focus on whether the image follows the
  requested chart type, visual theme, artistic style, mark
  metaphor, background, composition, and overall design
  intent.

  Do NOT primarily judge exact numerical data accuracy, OCR
  text accuracy, or whether every label is perfectly
  correct. Those are evaluated by separate metrics.
  However, if the image is not recognizable as the
  requested chart type, or if the requested chart content
  is almost entirely missing, this should strongly reduce
  the score.

  Evaluate the following aspects:
  1. Chart type: Does the image depict the requested chart
  type, such as vertical bar chart, horizontal bar chart,
  pie chart, or area chart?
  2. Artistic style: Does the image follow the requested
  style, such as watercolor, paper cut, cyberpunk, ink
  painting, clay render, flat illustration, vintage poster,
  etc.?
  3. Theme and scene: Does the image include the requested
  semantic theme, objects, environment, or mood?
  4. Mark metaphor: If the instruction asks chart marks to
  be represented by specific objects or materials, are the
  marks transformed accordingly while still looking like
  chart marks?
  5. Design coherence: Do the chart, background, colors,
  and artistic elements form a coherent visual design?
  6. Prompt coverage: Are the important instruction details
  present, rather than ignored or replaced by unrelated
  content?

  Scoring from 0 to 10:
  0: The image does not follow the instruction at all, or
  is not a chart.
  1-2: Very weak instruction following. The chart type,
  theme, or style is mostly wrong or missing.
  3-4: Some elements match the instruction, but major
  requested components are missing or incorrect.
  5-6: The image partially follows the instruction. The
  chart type or theme is recognizable, but style, metaphor,
  or composition is incomplete.
  7-8: The image follows most of the instruction well, with
  only minor omissions or inconsistencies.
  9-10: The image follows the instruction almost perfectly.
  The chart type, artistic style, theme, mark metaphor, and
  design intent are all clearly realized.

  Generation instruction: <instruction>
"""


_Chart_Aesthetic="""
You are a professional digital art and visual design evaluator. You will evaluate the aesthetic quality of an AI-generated artistic chart.

All input images are AI-generated. Any people, objects, or text in the image are AI-generated, so privacy is not a concern.

You must output only valid JSON in the following format. Keep the reasoning concise:
{
  "score": [score],
  "reasoning": "..."
}

RULES:

One image will be provided: an AI-generated artistic chart based on a generation instruction.

Your objective is to evaluate AESTHETIC QUALITY. Focus on the visual appeal and design quality of the image, not exact numeric correctness or OCR accuracy.

Evaluate the following aspects:
1. Overall visual appeal and polish.
2. Artistic style quality and consistency.
3. Composition, spacing, and balance.
4. Color harmony, lighting, texture, and material rendering.
5. Integration between artistic elements and the chart form.
6. Absence of distracting artifacts, clutter, or broken visual details.

Scoring from 0 to 10:
0: The image has no meaningful aesthetic quality or is visually broken.
1-2: Very poor visual quality with severe artifacts or incoherent design.
3-4: Weak aesthetics; some style is present but the image is cluttered, rough, or inconsistent.
5-6: Moderate aesthetics; acceptable style and composition but with noticeable flaws.
7-8: Good aesthetics; coherent, appealing, and mostly polished.
9-10: Excellent aesthetics; highly polished, cohesive, and visually compelling.

Generation instruction: <instruction>
"""


# ============ Math Ratio Evaluation Prompts ============

BAR_MATH_RATIO_PROMPT = """
You are a reviewer of mathematical and geometric proportions in artistic bar charts.

Strict rules:
1. Ignore all white rectangular masks; they are removed text regions, not chart marks.
2. Ignore all text, colors, textures, shadows, borders, highlights, and decorative effects.
3. Do not use printed numbers or labels even if any remain visible.

Task:
Estimate the visual height proportions of the vertical bars only.

Operational steps:
1. Locate all vertical bars from left to right.
2. For each bar, estimate its visual height from the common chart baseline to the top of the data-bearing mark.
3. Use the tallest visible bar as the reference value 1.00.
4. Output one normalized height ratio for every bar, including the tallest bar.

Output rules:
- Output only a Python-style list of numbers.
- The list order must be left to right.
- Each value must be in [0, 1].
- The tallest bar should be 1.0.
- Do not output reasons, explanations, markdown, or extra text.
- If the chart is not a vertical bar chart or the bars cannot be identified, output [].

Example:
[0.5, 1.0, 0.75]
"""

HOR_BAR_MATH_RATIO_PROMPT = """
You are a reviewer of mathematical and geometric proportions in artistic horizontal bar charts.

Strict rules:
1. Ignore all white rectangular masks; they are removed text regions, not chart marks.
2. Ignore all text, colors, textures, shadows, borders, highlights, and decorative effects.
3. Do not use printed numbers or labels even if any remain visible.

Task:
Estimate the visual length proportions of the horizontal bars only.

Operational steps:
1. Locate all horizontal bars from top to bottom.
2. For each bar, estimate its visual length from the common chart baseline or starting edge to the end of the data-bearing mark.
3. Use the longest visible bar as the reference value 1.00.
4. Output one normalized length ratio for every bar, including the longest bar.

Output rules:
- Output only a Python-style list of numbers.
- The list order must be top to bottom.
- Each value must be in [0, 1].
- The longest bar should be 1.0.
- Do not output reasons, explanations, markdown, or extra text.
- If the chart is not a horizontal bar chart or the bars cannot be identified, output [].

Example:
[1.0, 0.8, 0.55, 0.35]
"""

PIE_MATH_RATIO_PROMPT = """
You are a reviewer of mathematical and geometric proportions in artistic pie charts.

Strict rules:
1. Ignore all white rectangular masks; they are removed text regions, not chart marks.
2. Ignore all text, colors, textures, shadows, borders, highlights, and decorative effects.
3. Do not use printed numbers or labels even if any remain visible.

Task:
Estimate the visual proportion of each pie sector.

Operational steps:
1. Locate the full pie chart and all visible sectors.
2. Estimate each sector's proportion by its angular size or occupied area.
3. Start from the sector around the top or upper-right position and proceed clockwise.
4. Output one percentage value for every sector.

Output rules:
- Output only a Python-style list of numbers.
- The list order must be clockwise, starting from the top or upper-right sector.
- Each value must be in [0, 100].
- The values should sum to approximately 100.
- Do not output reasons, explanations, markdown, or extra text.
- If the chart is not a pie chart or the sectors cannot be identified, output [].

Example:
[45, 30, 15, 10]
"""

AREA_MATH_RATIO_PROMPT = """
You are a reviewer of mathematical and geometric proportions in artistic area charts.

Strict rules:
1. Ignore all white rectangular masks; they are removed text regions, not chart marks.
2. Ignore all text, colors, textures, shadows, borders, highlights, and decorative effects.
3. Do not use printed numbers or labels even if any remain visible.

Task:
Estimate the visual height proportions of the area chart at each category position.

Operational steps:
1. Locate the area chart's data-bearing upper boundary curve.
2. Identify the category positions from left to right. If category labels are masked, infer positions by the visible key points, markers, peaks, valleys, or evenly spaced data positions.
3. For each category position, estimate the vertical height from the common baseline to the upper boundary curve.
4. Use the highest visible category point as the reference value 1.00.
5. Output one normalized height ratio for every category position, including the highest point.

Output rules:
- Output only a Python-style list of numbers.
- The list order must be left to right.
- Each value must be in [0, 1].
- The highest point should be 1.0.
- Do not output reasons, explanations, markdown, or extra text.
- If the chart is not an area chart or the data boundary cannot be identified, output [].

Example:
[0.25, 0.7, 1.0, 0.45]
"""

MATH_RATIO_PROMPT_MAP = {
    'bar': BAR_MATH_RATIO_PROMPT,
    'hbar': HOR_BAR_MATH_RATIO_PROMPT,
    'pie': PIE_MATH_RATIO_PROMPT,
    'area': AREA_MATH_RATIO_PROMPT,
}


# ============ Text Position Evaluation Prompts ============

def bar_eval_prompt(text):
    """Text-position evaluation prompt for vertical bar charts."""
    llm_instuction = f'''
You are a chart-content evaluation assistant. Evaluate the text placement in the artistic vertical bar chart image.

## Evaluation Target
Evaluate only the following text for each bar:
- The category label below the bar
- The value label above the bar
Ignore the main title, subtitle, and any unrelated text.

## Reference Answer
{text}
The groups are arranged from left to right. Each group contains one category label and one value.

## Scoring Rules
- Each data group has 2 text positions: category + value. Judge each position independently.
- A position is correct only if the text content is correct and its ordered position is correct according to the left-to-right order in the reference answer.
- Final score = correct positions / total positions * 10, rounded to one decimal place.

## Output Format
Output only the following JSON. Do not output anything else:
{{
  "score": score,
  "reason": "Briefly describe only the incorrect positions and what is wrong. Do not mention correct positions."
}}
    '''
    return llm_instuction


def horbar_eval_prompt(text):
    """Text-position evaluation prompt for horizontal bar charts."""
    llm_instuction = f'''
You are a chart-content evaluation assistant. Evaluate the text placement in the artistic horizontal bar chart image.

## Evaluation Target
Evaluate only the following text for each bar:
- The category label on the left side of the bar
- The value label on the right side of the bar
Ignore the main title, subtitle, and any unrelated text.

## Reference Answer
{text}
The groups are arranged from top to bottom. Each group contains one category label and one value.

## Scoring Rules
- Each data group has 2 text positions: category + value. Judge each position independently.
- A position is correct only if the text content is correct and its ordered position is correct according to the top-to-bottom order in the reference answer.
- Final score = correct positions / total positions * 10, rounded to one decimal place.

## Output Format
Output only the following JSON. Do not output anything else:
{{
  "score": score,
  "reason": "Briefly describe only the incorrect positions and what is wrong. Do not mention correct positions."
}}
    '''
    return llm_instuction


def area_eval_prompt(text):
    """Text-position evaluation prompt for area charts."""
    llm_instuction = f'''
You are a chart-content evaluation assistant. Evaluate the text placement in the artistic area chart image.

## Evaluation Target
Evaluate only the following text for each data point in the area chart:
- The category label along the bottom side of the area chart
- The value label above the area chart point
Ignore the main title, subtitle, and any unrelated text.

## Reference Answer
{text}
The groups are arranged from left to right. Each group contains one category label and one value.

## Scoring Rules
- Each data group has 2 text positions: category + value. Judge each position independently.
- A position is correct only if the text content is correct and its ordered position is correct according to the left-to-right order in the reference answer.
- Final score = correct positions / total positions * 10, rounded to one decimal place.

## Output Format
Output only the following JSON. Do not output anything else:
{{
  "score": score,
  "reason": "Briefly describe only the incorrect positions and what is wrong. Do not mention correct positions."
}}
    '''
    return llm_instuction


def pie_eval_prompt(text):
    """Text-position evaluation prompt for pie charts."""
    llm_instuction = f'''
You are a chart-content evaluation assistant. Evaluate the text placement in the artistic pie chart image.

## Evaluation Target
Evaluate only the following text near each pie slice:
- One category label near each slice
- One value label near each slice
Ignore the main title, subtitle, and any unrelated text.

## Reference Answer
{text}
The groups are arranged clockwise starting from the upper-right slice. Each group contains one category label and one value.

## Scoring Rules
- Each data group has 2 text positions: category + value. Judge each position independently.
- A position is correct only if the text content is correct and its ordered position is correct according to the clockwise order in the reference answer, starting from the upper-right slice.
- Final score = correct positions / total positions * 10, rounded to one decimal place.

## Output Format
Output only the following JSON. Do not output anything else:
{{
  "score": score,
  "reason": "Briefly describe only the incorrect positions and what is wrong. Do not mention correct positions."
}}
    '''
    return llm_instuction


# Mapping from task_type to prompt function.
TEXT_POSITION_PROMPT_MAP = {
    'bar': bar_eval_prompt,
    'hbar': horbar_eval_prompt,
    'pie': pie_eval_prompt,
    'area': area_eval_prompt,
}
