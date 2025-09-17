# Argentine Legal Evolution Dataset - Codebook
## Variable Definitions and Coding Rules

### Version 1.0.0-beta
### Last Updated: September 2025

---

## Table of Contents

1. [Evolution Cases Dataset](#evolution-cases-dataset)
2. [Velocity Metrics Dataset](#velocity-metrics-dataset)
3. [Transplants Tracking Dataset](#transplants-tracking-dataset)
4. [Crisis Periods Dataset](#crisis-periods-dataset)
5. [Innovations Exported Dataset](#innovations-exported-dataset)
6. [Data Quality Standards](#data-quality-standards)
7. [Coding Conventions](#coding-conventions)

---

## Evolution Cases Dataset

### Primary Identifiers

**case_id** (String, Primary Key)
- Format: [3-letter area code][3-digit sequential number]
- Area codes: FID (Fideicomiso), CON (Constitutional), ADJ (Adjustment), PAG (Payment), AMP (Amparo), CAM (Exchange), LEA (Leasing), TRU (Trust), SEC (Securitization), CLA (Class Actions), COR (Corralito), PES (Pesification), FIN (Fintech), CRY (Crypto), COV (COVID), TEL (Telework), ENV (Environmental), IMP (Tax)
- Example: "FID001", "CON001", "ADJ001"

**nombre_caso** (String)
- Descriptive name in Spanish
- Maximum 50 characters
- Standard format: [Legal concept] [Specific type/modifier]
- Example: "Fideicomiso Financiero", "Convertibilidad Plena"

### Temporal Variables

**fecha_inicio** (Date, ISO 8601)
- Format: YYYY-MM-DD
- Date when legal norm/instrument was first introduced
- For gradual processes, use earliest significant milestone
- Missing values: Use "1900-01-01" for unknown

**fecha_fin** (Date, ISO 8601)
- Format: YYYY-MM-DD  
- Date when norm was formally repealed/replaced
- For ongoing processes: "2024-12-31"
- For failures: Date of effective abandonment

### Classification Variables

**area_derecho** (String, Controlled Vocabulary)
- Legal area classification
- Values: "Derecho Financiero", "Derecho Monetario", "Derecho Contractual", "Derecho Comercial", "Derecho Constitucional", "Derecho Cambiario", "Derecho Civil", "Derecho Bursatil", "Derecho Procesal", "Derecho Bancario", "Derecho Administrativo", "Derecho Laboral", "Derecho Ambiental", "Derecho Tributario"

**tipo_seleccion** (String, Controlled Vocabulary)
- Type of evolutionary selection mechanism
- Values: 
  - "Acumulativa": Gradual evolution building on previous norms
  - "Artificial": Deliberate top-down design/reform
  - "Mixta": Combination of both mechanisms

**origen** (String, Controlled Vocabulary)
- Origin of legal innovation
- Values:
  - "Endogeno": Developed within Argentine legal system
  - "Transplante": Direct adoption from foreign system
  - "Hibrido": Adaptation of foreign model to local conditions

**exito** (String, Controlled Vocabulary)
- Success level assessment
- Values:
  - "Exitoso": Achieved intended objectives, stable implementation
  - "Parcial": Mixed results, partial implementation
  - "Fracaso": Failed to achieve objectives, abandoned
  - "En_Desarrollo": Too early to assess (< 3 years)

### Quantitative Metrics

**velocidad_cambio_dias** (Integer)
- Days from introduction to end date
- For ongoing processes: Days from introduction to 2024-12-31
- Calculation: (fecha_fin - fecha_inicio) in days

**supervivencia_anos** (Integer)
- Years the norm survived in substantive form
- Rounded to nearest integer
- For ongoing: Years from introduction to 2024

**mutaciones_identificadas** (Integer)
- Count of significant modifications to original norm
- Only substantial changes (not minor technical adjustments)
- Range: 0-15+ (higher values indicate high mutation rate)

### Contextual Variables

**presion_ambiental** (String, Multiple Values Separated by Hyphen)
- Environmental pressures driving evolution
- Values: "Crisis", "Economica", "Tecnologica", "Social", "Politica", "Internacional"
- Format: Primary-Secondary (e.g., "Economica-Tecnologica")

**actores_principales** (String, Comma-Separated)
- Key institutional actors in evolution process
- Standard abbreviations: CNV (securities regulator), BCRA (central bank), CSJN (Supreme Court)
- Maximum 5 actors per case

**resistencias_documentadas** (String, Free Text)
- Description of documented opposition/resistance
- Maximum 100 characters
- Focus on institutional rather than individual resistance

### Diffusion Variables

**difusion_otras_jurisdicciones** (String, Comma-Separated)
- Countries that adopted similar norms
- Use standard country names in Spanish
- "Ninguna" if no diffusion identified

### Documentation Variables

**normativa_primaria** (String, Comma-Separated)
- Primary legal sources
- Format: [Type] [Number/Year] (e.g., "Ley 24.441", "Decreto 214/02")
- Maximum 5 sources per case

**fallos_relevantes** (String, Comma-Separated) 
- Key judicial decisions
- Format: [Court] [Case Name] (e.g., "CSJN Halabi")
- Use "No fallos relevantes" if none identified

**bibliografia_academica** (String, Author-Year Format)
- Key academic references
- Format: "Author (Year)" (e.g., "Lorenzetti (2003)")
- Maximum 3 references per case

**notas** (String, Free Text)
- Additional relevant information
- Maximum 200 characters
- Focus on unique characteristics or current status

---

## Velocity Metrics Dataset

### Metric Identification

**metric_id** (String, Primary Key)
- Format: [2-letter category][3-digit sequential]
- Categories: VEL (General velocity), VCC (Civil Code), VCM (Commercial), VTR (Tax), VFI (Financial), VCR (Crisis Response), VCA (Constitutional Amparo), VBC (Banking/Central Bank), VLA (Labor), VAD (Administrative), VPE (Procedural), VEN (Environmental), VCO (Consumer), VIM (Real Estate), VCY (Crypto)

**period** (String)
- Time period for measurement
- Format: "YYYY-YYYY" for ranges, "YYYY" for single years
- Standard periods: decade ranges (1950-1960, 1961-1970, etc.)

**area_derecho** (String)
- Legal area (same vocabulary as evolution_cases)
- "General" for cross-cutting metrics

**metric_type** (String, Controlled Vocabulary)
- Type of velocity measurement
- Values: "Reform_Frequency", "Codigo_Stability", "Major_Reforms", "Codigo_Lifespan", "Special_Laws", "Reform_Cycle", "Innovation_Speed", "Legal_Emergency_Speed", "Amparo_Cases", etc.

### Quantitative Values

**value** (Float)
- Numerical value of the metric
- Units specified in separate field
- Use null for missing values

**unit** (String)
- Unit of measurement
- Standard units: "reforms_per_year", "years_no_major_reform", "number_reforms", "months_avg_between_reforms", "days_innovation_to_regulation", "cases_per_year_avg"

### Comparative Context

**baseline_comparison** (String)
- Comparative benchmark
- Format: "[Country/Region]_[value]_[unit]" (e.g., "OECD_avg_1.8")
- Use internationally recognized benchmarks when available

**methodology** (String)
- Brief description of measurement methodology
- Maximum 50 characters
- Focus on data source and calculation method

**confidence_level** (String, Controlled Vocabulary)
- Assessment of data reliability
- Values: "High", "Medium", "Low"
- Based on source quality and measurement precision

---

## Transplants Tracking Dataset

### Transplant Identification

**transplant_id** (String, Primary Key)
- Format: "TR" + 3-digit sequential number
- Example: "TR001", "TR002"

**instrument_name** (String)
- Name of legal instrument being transplanted
- Use original language name with Spanish description if needed
- Maximum 100 characters

**origin_country** (String)
- Country of origin for the legal instrument
- Use standard country names
- For multi-country origins: list primary origin

**origin_legal_family** (String, Controlled Vocabulary)
- Legal system family of origin
- Values: "Common Law", "Civil Law", "Mixed", "Religious Law", "Socialist Law"

### Temporal Tracking

**introduction_date** (Date, ISO 8601)
- Date when transplant was first introduced in Argentina
- Format: YYYY-MM-DD

**modification_date** (Date, ISO 8601)
- Date of first significant modification
- Use "Never" for unmodified transplants
- Format: YYYY-MM-DD or "Never"

### Success Assessment

**success_level** (String, Controlled Vocabulary)
- Overall assessment of transplant success
- Values: "High", "Partial", "Failed", "Developing"

**adaptation_required** (String, Controlled Vocabulary)
- Level of adaptation needed for local context
- Values: "None", "Low", "Moderate", "Substantial", "Significant"

**local_resistance_level** (String, Controlled Vocabulary)
- Level of resistance encountered
- Values: "None", "Low", "Medium", "High", "Very High"

### Implementation Context

**institutional_sponsor** (String, Comma-Separated)
- Key institutions promoting the transplant
- Use standard abbreviations
- Maximum 5 sponsors

**market_reception** (String, Controlled Vocabulary)
- How market/users received the innovation
- Values: "Positive", "Mixed", "Negative"

### Diffusion Metrics

**survival_years** (Integer)
- Years the transplant survived in recognizable form
- For ongoing: years from introduction to 2024
- For failed: years until abandonment

**mutations_count** (Integer)
- Number of significant modifications
- Count substantial changes only
- Range: 0-10+

**export_potential** (String, Controlled Vocabulary)
- Potential for re-export to other countries
- Values: "High", "Medium", "Low", "None"

---

## Crisis Periods Dataset

### Crisis Identification

**crisis_id** (String, Primary Key)
- Format: [2-letter type][3-digit sequential]
- Types: CR (General Crisis), PR (Political), EC (Economic Sectoral), SC (Social), EN (Environmental), FI (Financial), IN (Infrastructure/Technology)

**crisis_name** (String)
- Descriptive name of crisis
- Maximum 50 characters
- Use commonly accepted historical names

**crisis_type** (String, Controlled Vocabulary)
- Type of crisis
- Values: "Monetary-Fiscal", "Financial-External", "Systemic-Multiple", "Political-Economic", "Health-Economic", "Environmental-Climate", etc.

**severity_level** (String, Controlled Vocabulary)
- Crisis severity assessment
- Values: "Low", "Medium", "High", "Very High", "Extreme"

### Legal Impact Metrics

**legal_changes_count** (Integer)
- Total number of legal changes during crisis
- Include laws, decrees, and significant regulations
- Range: 0-500+

**emergency_decrees** (Integer)
- Number of emergency decrees issued
- Constitutional emergency powers (DNU)
- Range: 0-100+

**new_laws** (Integer)
- Number of new laws passed by Congress
- Substantial new legislation only
- Range: 0-50+

**regulatory_changes** (Integer)
- Number of regulatory modifications
- Administrative rules, agency changes
- Range: 0-1000+

**institutional_reforms** (Integer)
- Number of institutional reforms
- Creation/elimination of agencies, structural changes
- Range: 0-50+

### Evolution Dynamics

**acceleration_factor** (Float)
- Multiplier of normal legal change velocity
- Calculation: crisis_period_velocity / normal_period_velocity
- Range: 1.0-10.0+

**pre_crisis_legal_stability** (String, Controlled Vocabulary)
- Legal system stability before crisis
- Values: "Low", "Medium", "High"

**post_crisis_innovations** (String, Free Text)
- Key innovations emerged from crisis
- Maximum 200 characters
- Focus on lasting institutional changes

### Contextual Indicators

**economic_indicators** (String, Free Text)
- Key economic metrics during crisis
- Format: "Indicator value, Indicator value"
- Maximum 100 characters

**political_stability** (String, Controlled Vocabulary)  
- Political system stability during crisis
- Values: "Very Low", "Low", "Medium", "High"

**international_pressure** (String, Free Text)
- International actors/pressures involved
- Maximum 100 characters
- Focus on institutional actors (IMF, World Bank, etc.)

**recovery_timeline_months** (Integer/Float)
- Months from crisis peak to recovery start
- Use decimals for sub-monthly precision
- "Ongoing" for unresolved crises

---

## Innovations Exported Dataset

### Innovation Identification

**innovation_id** (String, Primary Key)
- Format: "IN" + 3-digit sequential number
- Example: "IN001", "IN002"

**innovation_name** (String)
- Name of Argentine legal innovation
- Maximum 100 characters
- Use descriptive Spanish names

**origin_date** (Date, ISO 8601)
- Date innovation first appeared in Argentina
- Format: YYYY-MM-DD

**legal_area** (String)
- Legal area of innovation (consistent with other datasets)

**innovation_type** (String, Free Text)
- Type/category of innovation
- Maximum 100 characters
- Descriptive categorization

### Diffusion Tracking

**adopting_countries** (String, Comma-Separated)
- Countries that adopted the innovation
- Use standard country names
- Order by adoption date

**adoption_dates** (String, Comma-Separated)
- Dates of adoption in each country
- Format: YYYY-MM-DD for each date
- Order corresponding to adopting_countries

**adaptation_level** (String, Controlled Vocabulary)
- Level of adaptation required in adopting countries
- Values: "None", "Low", "Moderate", "Substantial", "High"

**success_level** (String, Controlled Vocabulary)
- Success of innovation in adopting countries
- Values: "Very High", "High", "Medium", "Low"

### Recognition Metrics

**diffusion_mechanism** (String, Free Text)
- How innovation spread internationally
- Maximum 100 characters
- Focus on transmission channels

**international_recognition** (String, Controlled Vocabulary)
- Level of international recognition
- Values: "Very High", "High", "Medium", "Low"

**documentation_level** (String, Controlled Vocabulary)
- Quality of documentation available
- Values: "Extensive", "Moderate", "Limited"

**academic_coverage** (String, Controlled Vocabulary)
- Level of academic study/coverage
- Values: "Very High", "High", "Medium", "Low"

**institutional_support** (String, Free Text)
- International institutions supporting diffusion
- Maximum 100 characters
- Focus on formal organizations

**regional_influence** (String, Controlled Vocabulary)
- Level of regional influence/adoption
- Values: "Very High", "High", "Medium", "Low"

---

## Data Quality Standards

### Missing Values
- Use standardized codes for missing data:
  - "Unknown": Information exists but not available
  - "N/A": Not applicable to this case
  - "Ongoing": Process still developing
  - null: Data not collected

### Date Formatting
- All dates in ISO 8601 format (YYYY-MM-DD)
- Use "1900-01-01" for unknown historical dates
- Use "2024-12-31" for ongoing processes
- Precision: day level preferred, month level acceptable

### Text Fields
- Maximum character limits enforced
- UTF-8 encoding
- Spanish language for descriptive fields
- Standard abbreviations documented separately

### Numerical Precision
- Integers for counts and years
- Floats (2 decimal places) for rates and ratios
- Consistent units within variables
- Range validation rules applied

---

## Coding Conventions

### Identifier Patterns
- Consistent format across datasets
- Meaningful prefixes for categorization
- Sequential numbering within categories
- No spaces or special characters (except hyphens)

### Controlled Vocabularies
- Predefined value lists for classification variables
- Spanish terms for legal concepts
- English for technical/international terms
- Regular vocabulary review and updates

### Temporal Consistency
- All time periods consistently defined
- Overlapping periods clearly documented
- Crisis periods vs. normal periods distinguished
- Historical context preserved

### Source Attribution
- Primary sources cited in standardized format
- Academic sources in Author (Year) format
- Legal sources with official identification
- International sources with organization attribution

---

## Version Control

### Data Updates
- Version numbering: Major.Minor.Patch
- Change log maintained separately
- Backward compatibility preserved when possible
- Deprecated fields marked clearly

### Quality Control
- Regular validation checks
- Cross-reference verification
- Expert review process
- Community feedback incorporation

---

**Document Status**: Version 1.0.0-beta  
**Last Review**: September 2025  
**Next Review**: December 2025  
**Maintainer**: Adrian Lerer  
**Contact**: [Repository maintainer contact]