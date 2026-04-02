n distributional semantics, Typical (also known as Syntactic or Substitutionary) relatedness and Topical (also known as Semantic or Associative) relatedness are two distinct ways words can be "close" in meaning.

Topical (Associative) Relatedness: Words belong to the same subject matter or semantic field. They co-occur often in the same document but might not be replaceable.

Examples: doctor & hospital, coffee & mug, sun & beach.

Typical (Substitutionary/Functional) Relatedness: Words share semantic features and function similarly in a sentence. They can often replace each other in a sentence without changing the grammatical structure.

Examples: car & bus (both are vehicles), big & huge (synonyms), run & walk (actions).


graph TD
    %% Main Title
    classDef title fill:none,stroke:none,font-size:18px,font-weight:bold;
    Title(Semantic Relatedness in Embedding Models):::title
    
    %% Split into Typical vs Topical
    subgraph "Natures of Relatedness"
        direction LR
        Typical[<b>Typical Relatedness</b><br/>'(Functional / Substitutionary)'<br/><br/>Words have high feature overlap<br/>They can often replace each other<br/>in similar local contexts.<br/><br/><i>Examples: car ↔ bus, run ↔ walk, huge ↔ big</i>]
        Topical[<b>Topical Relatedness</b><br/>'(Associative / Semantic Field)'<br/><br/>Words share a common subject<br/>They co-occur frequently<br/>within the same whole documents.<br/><br/><i>Examples: doctor ↔ hospital, coffee ↔ mug, sun ↔ beach</i>]
    end
    
    Typical --- Topical
    
    %% Connecting Models based on priority
    Typical ==>|Strongly Prioritizes| W2V[<b>Word2Vec</b><br/>'(Skip-gram / CBOW)'<br/><br/><b>Mechanism:</b><br/>Uses a <i>local sliding window</i><br/>(e.g., 5 words) to predict nearby context.<br/><br/><b>Result:</b><br/>Learns words that appear next to<br/>the same context words.]
    
    Topical ==>|Strongly Prioritizes| LSA[<b>LSA</b><br/>'(Latent Semantic Analysis)'<br/><br/><b>Mechanism:</b><br/>Uses <i>global document-level</i><br/>co-occurrence (SVD on Term-Doc Matrix).<br/><br/><b>Result:</b><br/>Learns words that appear in<br/>the same overall documents.]

    Typical -.->|Good Balance| GloVe[<b>GloVe</b><br/>'(Global Vectors)'<br/><br/><b>Mechanism:</b><br/>Factorizes a global <i>word-word</i><br/>co-occurrence matrix (counts how often<br/>words appear near each other globally).<br/><br/><b>Result:</b><br/>Strong on analogies (Typical) while<br/>still capturing topic (Topical).]
    Topical -.->|Good Balance| GloVe

    %% Styling
    classDef model box fill:#f9f,stroke:#333,stroke-width:2px,color:black;
    classDef type box fill:#e1f5fe,stroke:#01579b,stroke-width:1px,color:black;
    class W2V,LSA,GloVe model;
    class Typical,Topical type;