import csv

results = []
with open('/home/barshikar.s/depth-aware-dehazing/outputs/per_image_analysis.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        filename = row['filename']
        diff = float(row['diff'])
        
        parts = filename.replace('.jpg', '').split('_')
        if len(parts) >= 3:
            try:
                beta = float(parts[1])
                A = float(parts[2])
                results.append({'filename': filename, 'beta': beta, 'A': A, 'diff': diff})
            except:
                pass

groups = {0.08: [], 0.12: [], 0.16: [], 0.2: []}
for r in results:
    A = r['A']
    if A in groups:
        groups[A].append(r['diff'])

print("Depth improvement by haze density")
print(f"{'Scattering (A)':<15} {'Avg Diff (dB)':<15} {'Improved':<10} {'Degraded':<10}")

for A in sorted(groups.keys()):
    diffs = groups[A]
    avg = sum(diffs) / len(diffs)
    improved = sum(1 for d in diffs if d > 0.5)
    degraded = sum(1 for d in diffs if d < -0.5)
    print(f"{A:<15} {avg:+.2f}           {improved:<10} {degraded:<10}")

print("\nConclusion:")
print("  A = 0.08 (light haze): Depth HURTS")
print("  A = 0.16-0.2 (dense haze): Depth HELPS")