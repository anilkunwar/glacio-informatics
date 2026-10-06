To effectively hunt for `Rate Factor 1` and `Rate Factor 2`, the explorer code requires significant upgrades. Unlike the Glen exponent (which is just a simple integer like $n=3$), Rate Factors are massive scientific notation values that are **lethal to modelers if the units are wrong**. 

A value of `1.258e13` is only correct for $a^{-1} Pa^{-3}$. If a paper publishes the rate factor in $s^{-1} MPa^{-3}$, the value will be drastically different (e.g., $3.5e-25$). If you extract the number without extracting the **units**, the data is dangerous.

Here are the 4 structural changes you must make to the explorer code to safely harvest Rate Factors:

### 1. Update the Domain Vocabularies
Add terms that specifically target the Arrhenius parameterization of the flow law.

```python
# Add to your domain term sets
RATE_FACTOR_TERMS = {
    'rate factor', 'pre-exponential factor', 'pre-exponential', 
    'arrhenius factor', 'flow law parameter a', 'creep parameter',
    'arrhenius', 'activation energy', 'temperature limit', 
    'rate factor 1', 'rate factor 2', 'cold ice', 'temperate ice'
}

# Update the union
DOMAIN_TERMS = (GLEN_RHEOLOGY_TERMS | ICE_PHYSICS_TERMS | GLACIER_TERMS | RATE_FACTOR_TERMS)
```

### 2. Create a Unit-Aware Extractor
This is the core modification. The regex must capture the scientific notation *and* the surrounding unit string (Pa, MPa, s, a, yr). It must also handle PyMuPDF's tendency to flatten $10^{-25}$ into `10-25`.

Add this function alongside `extract_candidate_glen_exponent`:

```python
def extract_candidate_rate_factor(full_text: str, max_hits: int = 15) -> str:
    """
    Candidate Rate Factor (A) snippets. Extracts scientific notation AND units.
    Flags unit mismatches against the SIF file's expected a-Pa-m system.
    """
    if not full_text or full_text.startswith("Error"):
        return ""
    
    t = _normalize_flattened_text(full_text)
    
    # Keywords indicating the rate factor
    kw = re.compile(
        r'rate\s+factor|pre[- ]?exponential|arrhenius\s+factor|'
        r'flow\s+law\s+parameter\s+a|creep\s+parameter|a\s*=\s*\d',
        re.IGNORECASE)
    
    # Regex for Scientific Notation (handling flattened PDFs)
    # Matches: 1.258e13, 1.258 x 10^13, 1.258x1013, 3.5e-25, 3.5 10 -25
    sci_not = re.compile(
        r'\d+\.\d+\s*(?:e|E|x\s*10|xX)\s*[-−]?\s*\d+'
        r'|\d+\.\d+\s*10\s*[-−]?\s*\d+', 
        re.IGNORECASE)
    
    # Regex for Units (The critical safety net)
    # Matches: Pa-3 a-1, MPa-3 s-1, yr-1, a-1
    unit_pattern = re.compile(
        r'(?:M?Pa|kPa)\s*[-−]?\s*\d*\s*(?:a|s|yr|sec)\s*[-−]?\s*\d*',
        re.IGNORECASE)
    
    hits, seen = [], set()
    for m in kw.finditer(t):
        # Use a larger window to catch units that might be slightly separated
        window = re.sub(r'\s+', ' ', t[max(0, m.start() - 120): m.end() + 250]).strip()
        
        vals = sci_not.findall(window)
        units = unit_pattern.findall(window)
        
        if vals and window[:60] not in seen:
            seen.add(window[:60])
            
            # SIF UNIT VALIDATION (Elmer expects a-Pa-m)
            unit_warning = ""
            if units:
                unit_str = units[0]
                if 'MPa' in unit_str or 'Mpa' in unit_str:
                    unit_warning = " ⚠️ MPA_UNIT (SIF needs Pa)"
                elif 's-1' in unit_str or 'sec' in unit_str:
                    unit_warning = " ⚠️ SECONDS_UNIT (SIF needs a/yr)"
            
            hit_str = f"{window[:200]}  [→ {vals[0].strip()} {units[0].strip() if units else 'NO_UNIT'}{unit_warning}]"
            hits.append(hit_str)
            
        if len(hits) >= max_hits:
            break
            
    return " || ".join(hits)
```

### 3. Implement Regime Separation (Cold vs Warm Ice)
Because your SIF file splits the rate factor at `-10°C` (`Limit Temperature = Real -10.0`), you need a classifier that tags the extracted value as `Cold Ice (Rate Factor 1)` or `Warm Ice (Rate Factor 2)`.

```python
def classify_ice_regime(window_text: str) -> str:
    """Classifies if the extracted value belongs to cold or warm ice regime."""
    t = window_text.lower()
    # Warm ice indicators
    if any(kw in t for kw in ['temperate', 'warm ice', 't > -10', 'above -10', 'melting point']):
        return "WARM_ICE (Rate Factor 2)"
    # Cold ice indicators
    if any(kw in t for kw in ['cold ice', 't < -10', 'below -10', 'polar', 'interior']):
        return "COLD_ICE (Rate Factor 1)"
    return "UNSPECIFIED_REGIME"
```
*(You would call this inside the loop of `extract_candidate_rate_factor` and append it to the hit string).*

### 4. Update the Scopus Export Logic
Finally, you need to add the new extractor to your `export_scopus_format` function so the CSV/JSON includes the rate factor data alongside the Glen exponent data.

In the `export_scopus_format` function, modify the row dictionary:

```python
            row = {
                # ... existing Scopus columns ...
                
                # Replace the old single-extractor columns:
                "Candidate Glen n": extract_candidate_glen_exponent(full_text),
                "Candidate Rate Factor": extract_candidate_rate_factor(full_text),
                
                # ... rest of extra columns ...
            }
```

### Why this structural change is critical:
If you just extract `1.258e13` without the unit-aware NER, a user might look at a paper that states $A = 3.5 \times 10^{-25} \ s^{-1} MPa^{-3}$ and think "My SIF has `1.258e13`, this paper is completely wrong." 

By structurally enforcing unit extraction and ⚠️ warning flags, the explorer code transforms from a simple text scraper into a **dimensional analysis assistant**, explicitly telling the modeler: *"This paper uses MPa and seconds. You must mathematically convert this before putting it in your SIF."*
