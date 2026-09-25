import ast, datetime, hashlib, os, shutil, subprocess, sys

BASE = os.environ.get('GA_BASE', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OLD_MD5 = os.environ.get('EXPECT_MD5', '1e8b095eb0b97359b8b268cf40cc12a4')
TRAIN = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'train.py')
SUBMIT = ['submit_mlpga.sh', 'submit_fixed_grid.sh', 'submit_ops.sh', 'submit_ngold.sh']
VIEW = ['results.py', 'compare_baselines.py', 'new_results.py']

RULES = [
 ("argparse --variant",
  "'concat', 'perm', 'rand', 'noposition'])",
  "'concat', 'perm', 'rand', 'noposition', 'cnn_perm'])"),
 ("argparse --gnn",
  "choices=['pna', 'gcn', 'gat', 'sage', 'gin', 'nograph'])",
  "choices=['pna', 'gcn', 'gat', 'sage', 'gin', 'nograph', 'mlp'])"),
 ("USES_GATE",
  "USES_GATE = {'stat', 'cnn', 'gate_only', 'perm', 'rand', 'noposition'}",
  "USES_GATE = {'stat', 'cnn', 'gate_only', 'perm', 'rand', 'noposition', 'cnn_perm'}"),
 ("USES_PROJ",
  "USES_PROJ = {'stat', 'cnn', 'proj_only', 'perm', 'rand', 'noposition'}",
  "USES_PROJ = {'stat', 'cnn', 'proj_only', 'perm', 'rand', 'noposition', 'cnn_perm'}"),
 ("build_seq cnn_perm",
  "    if variant == 'cnn':\n"
  "        return torch.tensor(onehot_features(seqs)), f'one-hot [{len(seqs)}, {SEQ_LEN}, 4]'\n",
  "    if variant == 'cnn':\n"
  "        return torch.tensor(onehot_features(seqs)), f'one-hot [{len(seqs)}, {SEQ_LEN}, 4]'\n"
  "    if variant == 'cnn_perm':\n"
  "        # Permutation control for the CNN pathway: every one-hot sequence stays\n"
  "        # intact, but which CpG it belongs to is destroyed.  Same seed as 'perm'.\n"
  "        oh = onehot_features(seqs)\n"
  "        rng = np.random.default_rng(10_000 + seed)\n"
  "        return (torch.tensor(oh[rng.permutation(len(oh))]),\n"
  "                f'one-hot, row-permuted across CpGs [{len(seqs)}, {SEQ_LEN}, 4]')\n"),
 ("Net.__init__ CNN encoder",
  "        if variant == 'cnn':\n            self.seq_cnn = nn.Sequential(",
  "        if variant in ('cnn', 'cnn_perm'):\n            self.seq_cnn = nn.Sequential("),
 ("Net.__init__ gnn mlp",
  "        elif gnn == 'nograph':\n            self.conv = nn.Linear(self.in_dim, 1)\n",
  "        elif gnn == 'nograph':\n            self.conv = nn.Linear(self.in_dim, 1)\n"
  "        elif gnn == 'mlp':\n"
  "            # no graph and no node attributes: the readout MLP sees raw beta only\n"
  "            self.conv = None\n"),
 ("encode_seq",
  "        if self.variant != 'cnn':\n            return x_seq",
  "        if self.variant not in ('cnn', 'cnn_perm'):\n            return x_seq"),
 ("forward mlp",
  "        elif self.gnn == 'nograph':\n            x = self.conv(x)\n",
  "        elif self.gnn == 'nograph':\n            x = self.conv(x)\n"
  "        elif self.gnn == 'mlp':\n"
  "            x = x[:, 0:1]\n"),
]

def md5(p): return hashlib.md5(open(p, 'rb').read()).hexdigest()
def rd(p): return open(p, 'rb').read().decode('utf-8')
def wr(p, s, like=None):
    open(p, 'wb').write(s.encode('utf-8'))
    if like: shutil.copymode(like, p)
def die(msg): print('\n✗ ' + msg + '\n  No files were modified.'); sys.exit(1)

print('================ 1. Checks ================')
try:
    q = subprocess.run(['squeue', '-u', os.environ.get('USER', ''), '-h', '-o', '%j %T'],
                       capture_output=True, text=True, timeout=30)
    if q.returncode != 0: raise RuntimeError(q.stderr.strip())
    jobs = [l.split() for l in q.stdout.splitlines() if len(l.split()) == 2]
except Exception as e:
    die('could not run squeue: %s' % e)
refill = [n for n, s in jobs if n.startswith('ga_refill')]
if refill:
    die('A background refill job is still running (%s). Wait for it to print STOP and exit before patching.' % ', '.join(refill))
blocking = [n for n, s in jobs if n.startswith('ga_') and s == 'PENDING']
if blocking:
    die('%d GPU jobs are still queued (%s). They will read the new train.py when they start; wait until they are running or finished before patching.'
        % (len(blocking), ', '.join(blocking[:6])))
print('✓ no background refill job and no queued GPU jobs (running jobs already loaded the old code and are unaffected)')

if not os.path.exists(TRAIN): die('%s not found' % TRAIN)
cur = md5(TRAIN)
if cur != OLD_MD5: die('train.py has md5 %s, expected %s' % (cur, OLD_MD5))
print('✓ train.py md5 = %s (matches all completed experiments)' % cur)

src = rd(TRAIN)
new = src
try:
    ast.parse(src); can_parse = True
except SyntaxError:
    can_parse = False
for name, old, rep in RULES:
    n = new.count(old)
    if n != 1: die('anchor "%s" occurs %d times (must be exactly 1)' % (name, n))
    new = new.replace(old, rep)
    print('✓ anchor "%s" is unique; replaced' % name)
if can_parse:
    try:
        ast.parse(new)
    except SyntaxError as e:
        die('syntax error after patching: %s' % e)
    print('✓ patched file passes the syntax check')
else:
    print('! this Python cannot parse the original train.py (too old); skipping the syntax check')

subs = {}
for fn in SUBMIT:
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), fn)
    if not os.path.exists(p):
        if fn == 'submit_mlpga.sh': die('submit_mlpga.sh has not been created yet -- create it before patching')
        print('- %s not found, skipping' % fn); continue
    t = rd(p); k = 'MD5_EXPECT=' + OLD_MD5
    if t.count(k) != 1: die('%s: "%s" occurs %d times (must be exactly 1)' % (fn, k, t.count(k)))
    subs[p] = t

print('\n================ 2. Write ================')
bak = TRAIN + '.bak_' + OLD_MD5[:8]
if not os.path.exists(bak):
    wr(bak, src, TRAIN); print('✓ backed up the old version to %s' % os.path.relpath(bak, BASE))
tmp = TRAIN + '.tmp'
wr(tmp, new, TRAIN); os.replace(tmp, TRAIN)
NEW_MD5 = md5(TRAIN)
print('✓ train.py updated   new md5 = %s' % NEW_MD5)

for p, t in subs.items():
    tmp = p + '.tmp'; wr(tmp, t.replace('MD5_EXPECT=' + OLD_MD5, 'MD5_EXPECT=' + NEW_MD5), p); os.replace(tmp, p)
    print('✓ MD5_EXPECT updated in %s' % os.path.basename(p))

VIEW_RULES = [
 ('(none|cnn|stat)_fold', '(none|cnn_perm|cnn|stat)_fold'),
 ('|nograph)(.*)$', '|nograph|mlp)(.*)$'),
 ("('none', 'nograph'): 'nograph'}", "('none', 'nograph'): 'nograph', ('none', 'mlp'): 'MLP-GA', ('cnn_perm', 'pna'): 'CNN-perm'}"),
 ("('none','nograph'):'nograph'}", "('none','nograph'):'nograph',('none','mlp'):'MLP-GA',('cnn_perm','pna'):'CNN-perm'}"),
 ("order = ['none/PNA', 'CNN', 'nograph', 'stat']", "order = ['none/PNA', 'CNN', 'CNN-perm', 'nograph', 'MLP-GA', 'stat']"),
 ("ORDER=['none/PNA','CNN','nograph','stat']", "ORDER=['none/PNA','CNN','CNN-perm','nograph','MLP-GA','stat']"),
]
for fn in VIEW:
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), fn)
    if not os.path.exists(p): print('- %s not found, skipping' % fn); continue
    t = rd(p); n0 = 0
    for old, rep in VIEW_RULES:
        if old in t and rep not in t: t = t.replace(old, rep); n0 += 1
    try: ast.parse(t)
    except SyntaxError as e: print('! %s: syntax error after update, left unchanged: %s' % (fn, e)); continue
    wr(p, t); print('✓ %s now recognises MLP-GA / CNN-perm (%d replacements)' % (fn, n0))

log = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'TRAIN_PY_VERSIONS.txt')
with open(log, 'a', encoding='utf-8') as fh:
    fh.write('%s  %s -> %s  added --gnn mlp and --variant cnn_perm (additive; existing variants bit-identical)\n'
             % (datetime.datetime.now().strftime('%Y-%m-%d %H:%M'), OLD_MD5, NEW_MD5))
print('✓ version record appended to TRAIN_PY_VERSIONS.txt')
print('\nDone. New md5 = %s' % NEW_MD5)
