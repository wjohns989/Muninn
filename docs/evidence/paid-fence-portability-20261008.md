# Paid reload fence portability — October 8, 2026

Linux Tests run 37828702381, head 6525b5afd4b04c347e7a82e05159bf6aab065a70,
passed the entire affected portable-recovery group and clean-install job. The
full suite reached 2556 passes/27 skips before its first failure at
`test_known_submission_fences_policy_outbox_and_journal_until_exit` after
782.48 seconds, approximately 63% of the suite.

That test invoked the real Windows operator's `paid_stop_fence`, which opens an
archive using unattended Windows protection. Linux correctly required a
passphrase. This was another test-platform mismatch, not an authorization to
weaken the production reload or archive boundary.

The affected file now runs two explicit backends. Portable tests constrain the
archive to the fixture's temporary root and scope a constructor alias only
inside the real fence context, using the actual fixture phrase and actual
encrypted archive/authentication. The alias is restored on exit. Windows tests
keep the unmodified unattended constructor; only that backend is skipped on
Linux. No DPAPI success, outbox identity or fence decision is fabricated.

Original assertions remain: all three stores are writer-locked until fence
exit; bytes stay unchanged; locks release afterward; unknown synchronous/batch
submissions, wrong generation/repair bindings and tampered ciphertext cannot
reach stop; only the current paid batch is decrypted; the retained outbox is
included in a valid preimage. A new wrong-phrase case proves no fence yield and
unchanged databases. Production `scripts/reload_shared_local.py` is unchanged.

Initial portable checks used the unrelated cited-source fixture's phrase and
correctly failed authentication; the actual capture fixture was then inspected
and its known synthetic phrase used. Final combined affected validation:
130 Windows passes in 25.89 seconds, including 19 paid-fence checks. The new
Linux full-suite result is still pending; Windows proof does not substitute.

The paid-fence file joins the non-fail-fast affected portability preflight.
Locked dependencies, dummy tray backend, stack diagnostics and the required
full suite are retained. Only the full-suite job timeout changes from 20 to
35 minutes: the measured 782.48 seconds at 63% projects roughly 1242 seconds,
plus 128 seconds of preflight and setup. Test costs are uneven, so that is a
planning estimate, not measured completion time. Independent native review
cleared this bounded timing adjustment and the actual final test/workflow diff.
