# Authenticated dashboard navigation and read-only controls

## Observed behavior and repair

The shared content pane retained its previous scroll offset when switching pages.
An authenticated, GET/HEAD-only Edge probe reproduced the defect at 1280x900
and 390x844: the Encrypted capture status heading appeared at -1404 and
-3102.8125 pixels, respectively. DOM visibility alone had not detected this.

`showTab` now validates the destination before changing navigation state and
resets the shared content pane only when changing to a different valid page.
Refreshing the same page preserves reading position. No authentication,
provider, permission, dispatch, or backend lifecycle behavior changed.

The focused regression failed against the original function at `1567 != 0`.
After the repair, 41 dashboard control/accounting/policy/authentication tests
passed in 5.88 seconds. The same live browser probe then measured scrollTop 0
and heading positions 163 and 239.1875 pixels: both were in the viewport.
All browser requests were limited to the existing local service and GET/HEAD;
no write or external browser request occurred, and no service restart was used.

## Other bounded live proof from this installation session

Authenticated Edge checks matched all 11,377 displayed capture-lane jobs to
the backend and exercised eight keyboard navigation targets. Separate explicit
read controls displayed current batch permission generation 2, a 10,000-batch
quota with 9,960 remaining, the fixed-run accounting cutoff and $1.208825 ledger
total, one unknown admission, and the provider comparison's unreconciled-billing
warning. At 390 pixels there was no horizontal document overflow. Locking the
browser session cleared its accounting inputs and disabled permission saving.

These were read-only controls, not positive permission-save/revocation tests,
credential-use tests, transcript/citation retrieval proof, or full installation
acceptance. Authentication material was neither printed nor persisted.
