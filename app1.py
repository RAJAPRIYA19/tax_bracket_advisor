from pyswip import Prolog
import sys
import os

# Fix encoding to display ₹ on Windows
if os.name == 'nt':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        import ctypes
        ctypes.windll.kernel32.SetConsoleOutputCP(65001)

# --------------------------
# User Input
# --------------------------
try:
    age = int(input("Enter your age: "))
    income = int(input("Enter your annual income (₹): "))
except ValueError:
    print("❌ Invalid input. Please enter numbers.")
    exit()

# --------------------------
# Display Max Deduction Limits
# --------------------------
print("\n🔒 Maximum Deduction Limits for FY 2024-25:")
print(" - 80C (Investments, LIC, Tuition, etc.): ₹150,000")
print(f" - 80D (Medical Insurance): ₹{'50,000' if age >= 60 else '25,000'}")
print(" - 80CCD(1B) (NPS): ₹50,000")
print(" - EPF (Part of 80C): Included in ₹150,000")
print(" - NPS (Part of 80C): Included in ₹150,000")
print(" - LifeInsurance (Part of 80C): Included in ₹150,000")

# --------------------------
# Deduction Input
# --------------------------
deduction_input = input("\nEnter deductions (e.g., 80C=120000,80D=25000,80CCD(1B)=20000): ")
deduction_pairs = [d.strip() for d in deduction_input.split(",") if "=" in d]
deductions = {}

for pair in deduction_pairs:
    try:
        sec, amt = pair.split("=")
        deductions[sec.strip().upper()] = int(amt.strip())
    except ValueError:
        print(f"⚠️  Skipping invalid entry: {pair}")

# --------------------------
# Prolog Setup
# --------------------------
prolog = Prolog()
try:
    prolog.consult("tax_advisor.pl")
except Exception as e:
    print(f"❌ Failed to load Prolog file: {e}")
    exit()

# Clear old facts
for q in ["retractall(income(_))", "retractall(age(_))", "retractall(deduction(_, _))"]:
    list(prolog.query(q))

# Add new facts
prolog.assertz(f"income({income})")
prolog.assertz(f"age({age})")
for section, amount in deductions.items():
    # Enclose section in single quotes to support strings with special characters
    prolog.assertz(f"deduction('{section}', {amount})")

# --------------------------
# Output
# --------------------------
try:
    list(prolog.query("tax_summary."))  # Prolog will print everything from tax_summary/0
except Exception as e:
    print(f"❌ Error running tax_summary: {e}")

#chcp 65001
