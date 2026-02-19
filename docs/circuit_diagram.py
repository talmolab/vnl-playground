"""
Technical circuit diagram of the LRN-Cerebellum spiking policy.
Uses Graphviz for clean, automatic layout.
"""

import graphviz

g = graphviz.Digraph('LRN_Cerebellum',
                     format='png',
                     engine='dot')

g.attr(rankdir='TB', bgcolor='white', fontname='Helvetica',
       pad='0.5', nodesep='0.6', ranksep='0.8',
       label=('<<B><FONT POINT-SIZE="22">LRN–Cerebellum Spiking Motor Control Circuit</FONT></B>'
              '<BR/><FONT POINT-SIZE="11" COLOR="#666666"><I>'
              'Alstermark &amp; Ekerot (2015) · Wolpert &amp; Kawato (1998)</I></FONT>>'),
       labelloc='t', labeljust='c')

# Default node style
g.attr('node', fontname='Helvetica', fontsize='11', style='filled',
       penwidth='2')
g.attr('edge', fontname='Helvetica', fontsize='9')


# ═══════════════════════════════════════════════════════════════════════════════
# OBSERVATION INPUT
# ═══════════════════════════════════════════════════════════════════════════════

g.node('obs',
       label=('<<B>Sensory Observation</B>'
              '<BR/><FONT POINT-SIZE="9" COLOR="#555555"><I>'
              'proprioception + target (obs_dim)</I></FONT>>'),
       shape='box', style='filled,rounded', fillcolor='#B2EBF2',
       color='#00838F', penwidth='2.5')


# ═══════════════════════════════════════════════════════════════════════════════
# PROPRIOSPINAL C3-C4
# ═══════════════════════════════════════════════════════════════════════════════

with g.subgraph(name='cluster_ps') as ps:
    ps.attr(label=('<<B><FONT POINT-SIZE="14">ProprioSpinal C3-C4</FONT></B>'
                   '<BR/><FONT POINT-SIZE="9" COLOR="#555"><I>'
                   'LIF · Dale\'s law · heterogeneous τ<SUB>m</SUB> ∈ [1, 5] · '
                   'refractory · 8 micro-steps</I></FONT>>'),
            style='rounded,filled', fillcolor='#F5F5F5', color='#757575',
            penwidth='2', labeljust='l')

    ps.node('input_proj',
            label=('<<B>Input Projection</B>'
                   '<BR/><FONT POINT-SIZE="9">Dense(obs_dim → 512)</FONT>>'),
            shape='box', style='filled,rounded', fillcolor='#EEEEEE',
            color='#9E9E9E', penwidth='1.5')

    # Excitatory population
    ps.node('exc',
            label=('<<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0">'
                   '<TR><TD><B><FONT COLOR="#D32F2F" POINT-SIZE="12">Excitatory (E)</FONT></B></TD></TR>'
                   '<TR><TD><FONT POINT-SIZE="9">n<SUB>exc</SUB> = 410 (80%)</FONT></TD></TR>'
                   '<TR><TD>'
                   '<FONT COLOR="#D32F2F" POINT-SIZE="16">● ● ● ● ● ● ● ● ···</FONT>'
                   '</TD></TR>'
                   '<TR><TD><FONT POINT-SIZE="8" COLOR="#888"><I>LIF w/ surrogate gradient</I></FONT></TD></TR>'
                   '</TABLE>>'),
            shape='box', style='filled,rounded', fillcolor='#FFCDD2',
            color='#D32F2F', penwidth='2')

    # Inhibitory population
    ps.node('inh',
            label=('<<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0">'
                   '<TR><TD><B><FONT COLOR="#1976D2" POINT-SIZE="12">Inhibitory (I)</FONT></B></TD></TR>'
                   '<TR><TD><FONT POINT-SIZE="9">n<SUB>inh</SUB> = 102 (20%)</FONT></TD></TR>'
                   '<TR><TD>'
                   '<FONT COLOR="#1976D2" POINT-SIZE="16">● ● ● ● ● ···</FONT>'
                   '</TD></TR>'
                   '<TR><TD><FONT POINT-SIZE="8" COLOR="#888"><I>LIF w/ surrogate gradient</I></FONT></TD></TR>'
                   '</TABLE>>'),
            shape='box', style='filled,rounded', fillcolor='#BBDEFB',
            color='#1976D2', penwidth='2')

    # Same rank for E and I
    ps.attr('node')
    with ps.subgraph() as s:
        s.attr(rank='same')
        s.node('exc')
        s.node('inh')

# Obs → Input proj
g.edge('obs', 'input_proj', color='#00838F', penwidth='2')

# Input proj → E and I
g.edge('input_proj', 'exc', color='#9E9E9E', penwidth='1.5')
g.edge('input_proj', 'inh', color='#9E9E9E', penwidth='1.5')

# Lateral connections (Dale's law)
g.edge('exc', 'inh',
       label=('<<FONT COLOR="#D32F2F"><B>|W<SUB>EI</SUB>|</B> (+)</FONT>>'),
       color='#D32F2F', penwidth='1.5', style='bold',
       constraint='false')
g.edge('inh', 'exc',
       label=('<<FONT COLOR="#1976D2"><B>−|W<SUB>IE</SUB>|</B> (−)</FONT>>'),
       color='#1976D2', penwidth='1.5', style='bold',
       constraint='false')


# ═══════════════════════════════════════════════════════════════════════════════
# MOTOR READOUT (from E only)
# ═══════════════════════════════════════════════════════════════════════════════

g.node('motor_readout',
       label=('<<B>Motor Readout</B>'
              '<BR/><FONT POINT-SIZE="9">Dense(n<SUB>exc</SUB>=410 → 36)</FONT>>'),
       shape='box', style='filled,rounded', fillcolor='#E1BEE7',
       color='#7B1FA2', penwidth='2')

g.edge('exc', 'motor_readout',
       label=('<<FONT COLOR="#D32F2F"><B>E spike rates</B>'
              '<BR/>(410-dim)</FONT>>'),
       color='#D32F2F', penwidth='2')


# ═══════════════════════════════════════════════════════════════════════════════
# LRN RELAY
# ═══════════════════════════════════════════════════════════════════════════════

with g.subgraph(name='cluster_lrn') as lrn:
    lrn.attr(label=('<<B><FONT POINT-SIZE="13">LRN Relay</FONT></B>'
                    '<BR/><FONT POINT-SIZE="9" COLOR="#555"><I>'
                    'Lateral Reticular Nucleus · excitatory-only LIF · no lateral connections</I></FONT>>'),
             style='rounded,filled', fillcolor='#FFF3E0', color='#F57C00',
             penwidth='2')

    lrn.node('lrn_neurons',
             label=('<<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0">'
                    '<TR><TD><B><FONT COLOR="#F57C00" POINT-SIZE="11">Excitatory LIF</FONT></B></TD></TR>'
                    '<TR><TD><FONT POINT-SIZE="9">n<SUB>lrn</SUB> = 256</FONT></TD></TR>'
                    '<TR><TD>'
                    '<FONT COLOR="#F57C00" POINT-SIZE="16">● ● ● ● ● ● ● ···</FONT>'
                    '</TD></TR>'
                    '</TABLE>>'),
             shape='box', style='filled,rounded', fillcolor='#FFE0B2',
             color='#F57C00', penwidth='1.5')

# "Efference copy" invisible merger node
g.node('efference_merge', shape='point', width='0.1', height='0.1',
       color='#F57C00')

g.edge('exc', 'efference_merge',
       label=('<<FONT COLOR="#F57C00"><B>E+I spike rates</B>'
              '<BR/>(512-dim)</FONT>>'),
       color='#F57C00', penwidth='2')
g.edge('inh', 'efference_merge',
       color='#F57C00', penwidth='2', style='dashed')
g.edge('efference_merge', 'lrn_neurons',
       label='<<FONT COLOR="#F57C00"><I>efference copy</I></FONT>>',
       color='#F57C00', penwidth='2')


# ═══════════════════════════════════════════════════════════════════════════════
# CEREBELLUM (Kalman Filter)
# ═══════════════════════════════════════════════════════════════════════════════

with g.subgraph(name='cluster_cb') as cb:
    cb.attr(label=('<<B><FONT POINT-SIZE="14">Cerebellum</FONT></B>'
                   '<BR/><FONT POINT-SIZE="9" COLOR="#555"><I>'
                   'Differentiable Kalman Filter — sensory forward model '
                   '(Wolpert &amp; Kawato 1998)</I></FONT>'
                   '<BR/><FONT POINT-SIZE="8" COLOR="#388E3C">'
                   'Learnable: F (dynamics), B (motor→state), H (state→obs), '
                   'Q, R (noise), θ, γ (gate) · state_dim=64</FONT>>'),
            style='rounded,filled', fillcolor='#E8F5E9', color='#388E3C',
            penwidth='2.5', labeljust='l')

    # Predict
    cb.node('predict',
            label=('<<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0">'
                   '<TR><TD><B><FONT COLOR="#388E3C" POINT-SIZE="11">Predict</FONT></B></TD></TR>'
                   '<TR><TD><FONT POINT-SIZE="10">x̂<SUB>pred</SUB> = x̂ F<SUP>T</SUP>'
                   ' + u B<SUP>T</SUP></FONT></TD></TR>'
                   '<TR><TD><FONT POINT-SIZE="9" COLOR="#888">P<SUB>pred</SUB> = F P F<SUP>T</SUP> + Q</FONT></TD></TR>'
                   '</TABLE>>'),
            shape='box', style='filled,rounded', fillcolor='#C8E6C9',
            color='#388E3C', penwidth='1.5')

    # Innovation
    cb.node('innovation',
            label=('<<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0">'
                   '<TR><TD><B><FONT COLOR="#C62828" POINT-SIZE="11">Innovation (ε)</FONT></B></TD></TR>'
                   '<TR><TD><FONT POINT-SIZE="10">ε = z<SUB>obs</SUB> − x̂<SUB>pred</SUB> H<SUP>T</SUP></FONT></TD></TR>'
                   '<TR><TD><FONT POINT-SIZE="8" COLOR="#888"><I>sensory prediction error</I></FONT></TD></TR>'
                   '</TABLE>>'),
            shape='box', style='filled,rounded', fillcolor='#FFCDD2',
            color='#C62828', penwidth='1.5')

    # Update
    cb.node('update',
            label=('<<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0">'
                   '<TR><TD><B><FONT COLOR="#388E3C" POINT-SIZE="11">Update</FONT></B></TD></TR>'
                   '<TR><TD><FONT POINT-SIZE="10">x̂<SUB>new</SUB> = tanh(x̂<SUB>pred</SUB> + K ε)</FONT></TD></TR>'
                   '<TR><TD><FONT POINT-SIZE="9" COLOR="#888">P<SUB>new</SUB> = (I − KH) P<SUB>pred</SUB></FONT></TD></TR>'
                   '</TABLE>>'),
            shape='box', style='filled,rounded', fillcolor='#C8E6C9',
            color='#388E3C', penwidth='1.5')

    # Innovation Gate
    cb.node('gate',
            label=('<<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0">'
                   '<TR><TD><B><FONT COLOR="#388E3C" POINT-SIZE="11">Innovation Gate</FONT></B></TD></TR>'
                   '<TR><TD><FONT POINT-SIZE="10">g = σ(γ · (‖ε‖ − θ))</FONT></TD></TR>'
                   '<TR><TD><FONT POINT-SIZE="8" COLOR="#888"><I>≈0 normal · ≈1 perturbation</I></FONT></TD></TR>'
                   '</TABLE>>'),
            shape='box', style='filled,rounded', fillcolor='#C8E6C9',
            color='#2E7D32', penwidth='1.5')

    # DCN Correction
    cb.node('dcn',
            label=('<<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0">'
                   '<TR><TD><B><FONT COLOR="#388E3C" POINT-SIZE="11">DCN Correction</FONT></B></TD></TR>'
                   '<TR><TD><FONT POINT-SIZE="10">c = tanh(W[x̂<SUB>new</SUB>; W<SUB>ε</SUB>ε]) · g</FONT></TD></TR>'
                   '<TR><TD><FONT POINT-SIZE="8" COLOR="#888"><I>Deep Cerebellar Nuclei output</I></FONT></TD></TR>'
                   '</TABLE>>'),
            shape='box', style='filled,rounded', fillcolor='#A5D6A7',
            color='#2E7D32', penwidth='2')

    # Internal flow
    cb.edge('predict', 'innovation', color='#388E3C', penwidth='1.5')
    cb.edge('innovation', 'update', color='#C62828', penwidth='1.5',
            label=('<<FONT POINT-SIZE="8" COLOR="#C62828">K·ε</FONT>>'))
    cb.edge('innovation', 'gate', color='#C62828', penwidth='1.5',
            label=('<<FONT POINT-SIZE="8" COLOR="#C62828">‖ε‖</FONT>>'))
    cb.edge('innovation', 'dcn', color='#C62828', penwidth='1',
            style='dashed',
            label=('<<FONT POINT-SIZE="8" COLOR="#C62828">ε</FONT>>'))
    cb.edge('gate', 'dcn', color='#2E7D32', penwidth='1.5',
            label=('<<FONT POINT-SIZE="8" COLOR="#2E7D32">gate g</FONT>>'))
    cb.edge('update', 'dcn', color='#388E3C', penwidth='1',
            style='dashed',
            label=('<<FONT POINT-SIZE="8" COLOR="#388E3C">x̂<SUB>new</SUB></FONT>>'))

    # Recurrent carry
    cb.edge('update', 'predict',
            label=('<<FONT POINT-SIZE="8" COLOR="#9E9E9E"><I>carry (x̂, P)</I></FONT>>'),
            color='#9E9E9E', penwidth='1.2', style='dashed',
            constraint='false')


# LRN → Cerebellum (mossy fiber)
g.edge('lrn_neurons', 'predict',
       label=('<<FONT COLOR="#F57C00"><B>mossy fiber (u)</B></FONT>>'),
       color='#F57C00', penwidth='2.5')

# Obs → Innovation (sensory feedback)
g.edge('obs', 'innovation',
       label=('<<FONT COLOR="#00838F"><B>sensory feedback (z)</B></FONT>>'),
       color='#00838F', penwidth='2.5', style='bold')


# ═══════════════════════════════════════════════════════════════════════════════
# SUMMATION → MOTOR NEURONS → OUTPUT
# ═══════════════════════════════════════════════════════════════════════════════

g.node('sum',
       label='<<B><FONT POINT-SIZE="20">Σ</FONT></B>>',
       shape='circle', style='filled', fillcolor='white',
       color='#7B1FA2', penwidth='3', width='0.6', height='0.6',
       fixedsize='true')

g.edge('motor_readout', 'sum',
       label=('<<FONT COLOR="#7B1FA2"><B>raw motor command</B></FONT>>'),
       color='#7B1FA2', penwidth='2.5')

g.edge('dcn', 'sum',
       label=('<<FONT COLOR="#388E3C"><B>w · correction</B>'
              '<BR/><FONT POINT-SIZE="8">w = σ(w<SUB>raw</SUB>)</FONT></FONT>>'),
       color='#388E3C', penwidth='2.5')


# ═══════════════════════════════════════════════════════════════════════════════
# MOTOR NEURONS
# ═══════════════════════════════════════════════════════════════════════════════

with g.subgraph(name='cluster_mn') as mn:
    mn.attr(label=('<<B><FONT POINT-SIZE="14">Motor Neuron Pool</FONT></B>'
                   '<BR/><FONT POINT-SIZE="9" COLOR="#555555"><I>'
                   'Excitatory-only LIF · no lateral connections · '
                   'heterogeneous τ<SUB>m</SUB> · 8 micro-steps</I></FONT>>'),
            style='rounded,filled', fillcolor='#F3E5F5', color='#8E24AA',
            penwidth='2.5')

    mn.node('mn_neurons',
            label=('<<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0">'
                   '<TR><TD><B><FONT COLOR="#8E24AA" POINT-SIZE="12">Excitatory LIF</FONT></B></TD></TR>'
                   '<TR><TD><FONT POINT-SIZE="9">n<SUB>mn</SUB> = 128</FONT></TD></TR>'
                   '<TR><TD>'
                   '<FONT COLOR="#8E24AA" POINT-SIZE="16">● ● ● ● ● ● ● ● ···</FONT>'
                   '</TD></TR>'
                   '<TR><TD><FONT POINT-SIZE="8" COLOR="#888888"><I>LIF w/ surrogate gradient</I></FONT></TD></TR>'
                   '</TABLE>>'),
            shape='box', style='filled,rounded', fillcolor='#CE93D8',
            color='#8E24AA', penwidth='2')

g.edge('sum', 'mn_neurons',
       label=('<<FONT COLOR="#8E24AA"><B>pre-motor cmd</B>'
              '<BR/>(36-dim)</FONT>>'),
       color='#8E24AA', penwidth='2.5')

g.node('muscle_readout',
       label=('<<B>Muscle Readout</B>'
              '<BR/><FONT POINT-SIZE="9">Dense(n<SUB>mn</SUB>=128 → 36)</FONT>>'),
       shape='box', style='filled,rounded', fillcolor='#CE93D8',
       color='#8E24AA', penwidth='2')

g.edge('mn_neurons', 'muscle_readout',
       label=('<<FONT COLOR="#8E24AA"><B>motor spike rates</B>'
              '<BR/>(128-dim)</FONT>>'),
       color='#8E24AA', penwidth='2.5')

g.node('action',
       label=('<<B>Action (logits)</B>'
              '<BR/><FONT POINT-SIZE="9"><I>NormalTanh → muscle ctrl [0,1] × 9 muscles</I></FONT>>'),
       shape='box', style='filled,rounded', fillcolor='#E1BEE7',
       color='#7B1FA2', penwidth='2.5')

g.edge('muscle_readout', 'action', color='#7B1FA2', penwidth='2.5')


# ═══════════════════════════════════════════════════════════════════════════════
# EXTERNAL PERTURBATION
# ═══════════════════════════════════════════════════════════════════════════════

g.node('perturb',
       label=('<<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0">'
              '<TR><TD><B><FONT COLOR="#C62828" POINT-SIZE="11">External Force</FONT></B></TD></TR>'
              '<TR><TD><FONT POINT-SIZE="9">xfrc_applied on ulna</FONT></TD></TR>'
              '<TR><TD><FONT POINT-SIZE="8" COLOR="#888"><I>during [t<SUB>start</SUB>, t<SUB>end</SUB>)</I></FONT></TD></TR>'
              '</TABLE>>'),
       shape='box', style='filled,rounded,dashed', fillcolor='#FFEBEE',
       color='#C62828', penwidth='2')

g.edge('perturb', 'action',
       label=('<<FONT COLOR="#C62828" POINT-SIZE="8"><I>disrupts limb</I></FONT>>'),
       color='#C62828', penwidth='1.8', style='dashed')


# ═══════════════════════════════════════════════════════════════════════════════
# LIF DYNAMICS ANNOTATION
# ═══════════════════════════════════════════════════════════════════════════════

g.node('lif_note',
       label=('<<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="2">'
              '<TR><TD ALIGN="LEFT"><B><FONT POINT-SIZE="10">LIF Dynamics</FONT></B></TD></TR>'
              '<TR><TD ALIGN="LEFT"><FONT POINT-SIZE="9">α = exp(−dt/τ<SUB>m</SUB>)</FONT></TD></TR>'
              '<TR><TD ALIGN="LEFT"><FONT POINT-SIZE="9">v ← α·v + (1−α)·I</FONT></TD></TR>'
              '<TR><TD ALIGN="LEFT"><FONT POINT-SIZE="9">spike if v ≥ v<SUB>θ</SUB> = 0.3</FONT></TD></TR>'
              '<TR><TD ALIGN="LEFT"><FONT POINT-SIZE="8" COLOR="#888">surrogate: σ(β(v−v<SUB>θ</SUB>))</FONT></TD></TR>'
              '<TR><TD ALIGN="LEFT"><FONT POINT-SIZE="8" COLOR="#888">n<SUB>refrac</SUB> = 2</FONT></TD></TR>'
              '</TABLE>>'),
       shape='note', style='filled', fillcolor='#FFFDE7',
       color='#FBC02D', penwidth='1.5')

# Connect to PS cluster area invisibly for layout
g.edge('lif_note', 'exc', style='invis')


# ═══════════════════════════════════════════════════════════════════════════════
# RENDER
# ═══════════════════════════════════════════════════════════════════════════════

g.attr(dpi='200')
g.render('/root/vast/eric/vnl-playground/docs/circuit_diagram',
         cleanup=True)

# Also render PDF
g_pdf = g.copy()
g_pdf.format = 'pdf'
g_pdf.render('/root/vast/eric/vnl-playground/docs/circuit_diagram_vec',
             cleanup=True)

print("Saved: docs/circuit_diagram.png, docs/circuit_diagram_vec.pdf")
