# Keyword Clustering and Intent Tagger

Internal SEO tool for grouping search keywords into semantic clusters, classifying search intent, and matching clusters to existing articles on the website.

---

## What the Tool Does

1. **Uploads and Standardizes Keyword Exports**: Accepts `.csv` and `.xlsx` files from Ahrefs Free, Google Search Console, or custom spreadsheets with interactive column mapping.
2. **Cleans and Deduplicates**: Lowercases, strips extraneous whitespace, and removes duplicate keywords deterministically.
3. **Semantic Clustering**: Transforms keywords into sentence embeddings using `sentence-transformers` (`all-MiniLM-L6-v2`) and groups them using `AgglomerativeClustering` (cosine distance, average linkage) with a customizable distance threshold slider.
4. **Brand Guard and Niche Guard**:
   - **Brand Guard**: Reduces the chance of different brands merging into the same cluster by applying a distance penalty (+1.0) to conflicting single-brand pairs.
   - **Niche Guard**: Reduces the chance of different professional industries with shared generic patterns from merging.
5. **Rule-Based Search Intent Classification**: Assigns `informational`, `commercial`, `transactional`, or `navigational` intent to each keyword based on an editable modifier dictionary and brand list. Aggregates cluster-level intent (or marks as `mixed` if the dominant intent is below 60%).
6. **Article Matching**: Calculates cluster centroids, computes cosine similarity against existing articles' main keywords, and marks whether to update an existing page (`Добавить в существующую`) or write a new one (`Новая статья`), highlighting the second-best match and close calls.
7. **Two-Sheet Export**: Downloads an `.xlsx` workbook (with frozen header rows and auto-adjusted column widths) and `.csv` files encoded with UTF-8 BOM for Microsoft Excel compatibility.


## Input Format

- **Keyword File**: A `.csv` or `.xlsx` export file. Requires a keyword column; volume, difficulty (KD), clicks, and impressions columns are optional and mapped via dropdowns in the user interface.
- **Articles Text Area**: Pasted list of existing articles in the sidebar, one line per article in the format:
  `title | main_keyword | url`
- The default article list in the sidebar is set in `DEFAULT_ARTICLES` at the top of `app.py`.

---

## Output

The export produces an `.xlsx` workbook with two sheets, plus corresponding `.csv` downloads:

### 1. Sheet "Clusters"
- `cluster_id`: Numeric identifier of the cluster.
- `cluster_name`: Main keyword representing the cluster (highest volume, or shortest phrase if no volume).
- `cluster_intent`: Dominant search intent across cluster keywords (or `mixed` if the top intent is under 60%).
- `keywords_count`: Number of keywords assigned to the cluster.
- `total_volume`: Cumulative search volume of all keywords in the cluster.
- `avg_kd`: Average keyword difficulty across the cluster.
- `matched_article`: Title of the best-matching existing article (blank if below match threshold).
- `matched_url`: URL of the best-matching existing article.
- `similarity`: Cosine similarity score between the cluster centroid and the best-matching article's main keyword.
- `action`: Recommendation label (`Добавить в существующую` or `Новая статья`).
- `final_article`: Pre-filled with `matched_article` when action is `Добавить в существующую`, left blank for `Новая статья`; meant for manual review and editing in Excel.
- `second_match`: Title of the second most similar existing article.
- `second_similarity`: Cosine similarity score of the second-best match.
- `note`: Displays `check manually` if cluster intent is `mixed`, or (for matched clusters) if similarity is below 0.65, or if the gap between best and second-best similarity is below 0.05.
- `decision`: Empty column reserved for manual editorial review.
- `my_note`: Empty column reserved for specialist notes.

### 2. Sheet "Keywords"
- `keyword`: Cleaned keyword string.
- `volume`: Search volume value (0 if unmapped).
- `kd`: Keyword difficulty value (0 if unmapped).
- `cluster_id`: ID of the assigned cluster.
- `cluster_name`: Name of the assigned cluster.
- `intent`: Rule-based search intent (`informational`, `commercial`, `transactional`, or `navigational`).
- `decision`: Empty column reserved for manual review.
- `my_note`: Empty column reserved for specialist notes.
- *Sorting*: Sorted by `cluster_id` ascending, then `volume` descending.

---

## Installation & Running

### Requirements
- Python 3.x (check with `python --version`)
- Recommended: run within a virtual environment

### Quick Setup

1. Open a terminal in the project folder.

2. Create and activate a virtual environment (if not already created):
   ```bash
   python -m venv .venv
   # Windows PowerShell:
   .\.venv\Scripts\Activate.ps1
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Launch the Streamlit application:
   ```bash
   streamlit run app.py
   ```

5. Open `http://localhost:8501` in your browser. You can test using the provided `sample_keywords.csv` file.

---

## Tested On

### 1) Dataset: `sample_keywords.csv`
- 40 synthetic keywords, 9 real articles loaded in the sidebar.
- At cluster threshold 0.45 and match threshold 0.55: 21 clusters, 8 matched to existing articles, 13 marked as new article, 4 marked check manually.
- Intent distribution over 40 keywords: commercial 17, informational 13, transactional 8, navigational 2.
- Close call: 'ai contract review tools review' matched the lawyers article at 0.620, the second-best (real estate article) scored 0.616, so the choice is fragile and is flagged for manual review. 'how teachers can detect ai writing' matched the teachers tools article at 0.681 and is not flagged, but it may need its own article; matching compares only the main keyword of an article.

### 2) Dataset: `keyword_ideas_ahrefs.csv`
- 156 unique keyword ideas from Ahrefs Free Keyword Generator (first 20 ideas for each of 9 seed phrases, one seed duplicated and removed).
- At cluster threshold 0.45: 9 clusters (20, 20, 20, 20, 20, 20, 20, 9, 7 keywords), all 9 matched to existing articles with similarity from 0.936 to 0.978, 2 marked check manually (ai tools for lawyers, ai tools for architects).
- Intent over 156 keywords: informational 81, commercial 73, transactional 1, navigational 1.
- Note: The keywords are variants of my own article topics, so the 9 clusters match the 9 articles; this checks grouping and the niche guard, it does not find new topics. Ahrefs Free gives volume and KD only as categories (>1000, >100, <100; Easy/Medium/Hard), which I converted to rough numbers (1000/100/50; 15/45/75), so total_volume and avg_kd are not real figures on this dataset.

---

## Limitations

- **Semantic Similarity, Not SERP-Based Clustering**: The tool groups keywords based on vector embeddings in the `all-MiniLM-L6-v2` semantic space. It does not scrape live search engine results (SERP) or check URL overlap in Google Top 10.
- **Main Keyword Comparison Only**: Article matching compares the cluster with the article's main keyword only, so it does not know what sections a long article contains.
- **Rule-Based Intent Detection**: Search intent is assigned via dictionary lookups. Keywords without explicit trigger modifiers default to informational intent, and complex multi-intent phrases may require manual adjustment.
- **Informational Default for Head Phrases**: Keywords without modifier words default to informational, so head phrases like 'ai tools for architects' count as informational; a mixed label can come from this default.
- **Commercial Classification of 'Free'**: The word 'free' is treated as a commercial modifier, which is a choice and may not fit every topic.
- **Brand Intent Scope**: Brands inside longer phrases do not make a keyword navigational; only login/official site phrases or an exact brand name do.
- **Language of Action Labels**: Action labels in the table (`Добавить в существующую` and `Новая статья`) are in Russian.
- **Synthetic Test Data**: The sample file is synthetic, so results on real data may differ.
- **Brand Guard Scope**: Brand Guard only checks brands present in the predefined `BRANDS` list at the top of `app.py`. It does not detect unknown brands or brand typos.
- **Niche Guard Scope**: `NICHE_MARKERS` currently covers teachers, architects, lawyers, accountants and real estate and must be edited for other topics.
- **Threshold Calibration**: The clustering distance threshold (`0.45` default) and article matching threshold (`0.55` default) require adjustment depending on dataset breadth.
- **Human Review Required**: The export tables include empty `decision` and `my_note` columns, as automated clusters and article suggestions are meant to assist content briefs, not replace the SEO specialist's judgment.
